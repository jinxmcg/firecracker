// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-vCPU KVM dirty-ring support for low-stall live dirty-page export.
//!
//! # Why
//! Live precopy migration repeatedly asks KVM for the pages dirtied since the last
//! round. The classic mechanism (`KVM_GET_DIRTY_LOG`, bitmap mode) scans all guest
//! RAM and, crucially, **re-write-protects every dirty page** so the next pass can
//! detect new writes. On a busy guest that re-protect forces a TLB shootdown across
//! vCPUs followed by a write-fault storm, stalling the guest ~1-2s per round.
//!
//! The dirty ring (`KVM_CAP_DIRTY_LOG_RING[_ACQ_REL]`) instead logs each dirtied GFN
//! into a per-vCPU ring as it happens (via PML on Intel where available). Userspace
//! harvests the ring and issues a single `KVM_RESET_DIRTY_RINGS`, which re-protects
//! **only the harvested GFNs**, in bulk. This is the path QEMU uses for live
//! migration, and it cuts the per-round stall to near zero.
//!
//! # Correctness
//! A missed dirty page means silent guest-memory corruption on the destination, so the
//! harvest must lose nothing. Two subtleties are handled here:
//! * **`KVM_EXIT_DIRTY_RING_FULL`**: if a vCPU fills its ring mid-round it cannot make
//!   progress until userspace harvests. The vCPU thread therefore harvests into a
//!   shared accumulator (it never resets without first recording the GFNs), so the
//!   pages survive until the next export consumes them.
//! * **non-vCPU dirtying inside KVM**: pages dirtied outside a vCPU context (KVM-internal)
//!   are not logged to the ring. When `KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP` is available we
//!   enable it and additionally harvest the per-memslot bitmap, OR-ing it in. That cap is
//!   **ARM64-only** (gated on `CONFIG_HAVE_KVM_DIRTY_RING_WITH_BITMAP`, which arm64 selects
//!   and x86 does not); it exists for the GIC ITS, which x86 has no analogue of. So on x86
//!   `with_bitmap` is always false, and that is correct, not a probe failure — x86 KVM has
//!   no in-kernel non-vCPU dirtying that the ring misses.
//!
//! Note this is distinct from *device* dirtying (virtio DMA, queue rings): those writes
//! never touch KVM's MMU at all, so neither the bitmap nor the ring ever saw them. They are
//! tracked by Firecracker's own per-region `AtomicBitmap` and OR-ed into the export in
//! `dump_dirty` (`is_kvm_page_dirty || is_firecracker_page_dirty`), with the virtio queue
//! pages additionally pinned dirty by `mark_virtio_queues`. The ring switch changes only how
//! *vCPU* dirtying is reported and is therefore orthogonal to all device coverage.
//!
//! The harvest produces the same `slot -> Vec<u64>` bitmap currency the rest of the VMM
//! already consumes (`dump_dirty`), so only *how* the bitmap is produced changes.

use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use kvm_bindings::{
    KVM_CAP_DIRTY_LOG_RING, KVM_CAP_DIRTY_LOG_RING_ACQ_REL, KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP,
    KVM_DIRTY_LOG_PAGE_OFFSET, kvm_dirty_gfn, kvm_enable_cap,
};
use kvm_ioctls::{Kvm as KvmFd, VmFd};

use crate::arch::host_page_size;
use crate::logger::{info, warn};

/// Per the KVM uapi (`include/uapi/linux/kvm.h`), a ring entry is "dirty" (ready to be
/// harvested by userspace) when this flag is set in its `flags` word.
const KVM_DIRTY_GFN_F_DIRTY: u32 = 1 << 0;
/// Userspace sets this flag (clearing `F_DIRTY`) on a harvested entry; the subsequent
/// `KVM_RESET_DIRTY_RINGS` then re-protects exactly those GFNs.
const KVM_DIRTY_GFN_F_RESET: u32 = 1 << 1;

/// `KVM_RESET_DIRTY_RINGS = _IO(KVMIO, 0xc7)`. kvm-ioctls 0.24 has no first-class
/// dirty-ring API, so we issue this by hand. `KVMIO == 0xAE`; an `_IO` request (no
/// payload, no direction) encodes as `(type << 8) | nr`.
const KVM_RESET_DIRTY_RINGS: u64 = (0xAE << 8) | 0xc7;

/// Desired per-vCPU ring size in bytes, clamped down to the host maximum (see
/// [`clamp_ring_bytes`]). Each `kvm_dirty_gfn` is 16 bytes, so 1 MiB == 65536 entries
/// == 256 MiB worth of 4 KiB pages a vCPU can dirty before it must be harvested. Hosts
/// typically cap the ring at exactly this (`KVM_DIRTY_RING_MAX_ENTRIES == 65536`); we
/// request the largest the host allows to make `KVM_EXIT_DIRTY_RING_FULL` rare.
const DESIRED_RING_BYTES: usize = 1 << 20;

/// Errors that abort VM/vCPU creation when the dirty ring is in use. Once the ring cap
/// is enabled, KVM logs dirty pages into rings we *must* be able to read; failing to
/// map one would silently drop pages, so these are fatal rather than a fallback.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum DirtyRingError {
    /// Failed to mmap a vCPU dirty ring: {0}
    Mmap(std::io::Error),
}

/// The dirty-ring capability negotiated against the host.
#[derive(Debug, Clone, Copy)]
struct DirtyRingCaps {
    /// The ring cap to enable (`ACQ_REL` preferred, else the legacy `RING`).
    cap: u32,
    /// Per-vCPU ring size in bytes (power of two, multiple of page size).
    ring_bytes: usize,
    /// Whether `KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP` is also available.
    with_bitmap: bool,
}

/// A single vCPU's mmap'd dirty ring plus the harvest cursor into it.
#[derive(Debug)]
struct DirtyRing {
    /// mmap base: an array of `count` [`kvm_dirty_gfn`] shared with the kernel.
    base: *mut kvm_dirty_gfn,
    /// Number of entries in the ring (power of two).
    count: u32,
    /// Length of the mmap in bytes (for `munmap` on drop).
    mmap_len: usize,
    /// Monotonically increasing index of the next entry to harvest; indexed mod
    /// `count`. Persists across harvests, exactly as QEMU's `kvm_fetch_index`.
    fetch_index: u32,
}

// SAFETY: `base` points to a kernel mmap shared with the vCPU. Every access to a
// `DirtyRing` happens under the owning `Mutex<DirtyRingInner>`, and the per-entry
// `flags` word is read/written with acquire/release ordering to pair with the kernel's
// publication of each entry. The struct is therefore safe to send/share across threads.
unsafe impl Send for DirtyRing {}

impl DirtyRing {
    /// Drains every currently-dirty entry into `accum` (keyed by KVM memslot id),
    /// marking each consumed entry `RESET`. Returns `true` if anything was harvested
    /// (so the caller knows a `KVM_RESET_DIRTY_RINGS` is needed).
    fn harvest_into(&mut self, accum: &mut HashMap<u32, Vec<u64>>) -> bool {
        let start = self.fetch_index;
        let mut idx = start;
        loop {
            let slot_in_ring = (idx % self.count) as usize;
            // SAFETY: `slot_in_ring < count`, so this is in-bounds of the mapping.
            let entry = unsafe { self.base.add(slot_in_ring) };
            // SAFETY: `entry` is a valid, aligned `kvm_dirty_gfn` in the mapping; its
            // `flags` word is concurrently written by the kernel, so we only ever touch
            // it through atomic ops.
            let flags = unsafe { AtomicU32::from_ptr(std::ptr::addr_of_mut!((*entry).flags)) };
            // Acquire-load so the kernel's prior writes to `slot`/`offset` are visible.
            if flags.load(Ordering::Acquire) & KVM_DIRTY_GFN_F_DIRTY == 0 {
                break;
            }
            // SAFETY: published by the kernel before the F_DIRTY release-store we just
            // acquire-observed, so these plain reads see the intended values.
            let (slot, page) = unsafe {
                (
                    // low 16 bits: memslot id; high 16 bits: address-space id (0 for
                    // normal guest RAM, which is all Firecracker uses).
                    std::ptr::addr_of!((*entry).slot).read() & 0xffff,
                    // page index within the memslot (gfn - slot.base_gfn).
                    std::ptr::addr_of!((*entry).offset).read(),
                )
            };
            let word = (page / 64) as usize;
            let bit = page % 64;
            let v = accum.entry(slot).or_default();
            if v.len() <= word {
                v.resize(word + 1, 0);
            }
            v[word] |= 1u64 << bit;
            // Release-store RESET so KVM_RESET_DIRTY_RINGS re-protects this GFN.
            flags.store(KVM_DIRTY_GFN_F_RESET, Ordering::Release);
            idx = idx.wrapping_add(1);
        }
        self.fetch_index = idx;
        idx != start
    }
}

impl Drop for DirtyRing {
    fn drop(&mut self) {
        // SAFETY: `base`/`mmap_len` describe a mapping we created with mmap and have not
        // otherwise freed.
        unsafe {
            libc::munmap(self.base.cast(), self.mmap_len);
        }
    }
}

/// State guarded by the harvest mutex.
#[derive(Debug)]
struct DirtyRingInner {
    /// One ring per vCPU, in vCPU-index order.
    rings: Vec<DirtyRing>,
    /// Pages dirtied by vCPUs since the last [`DirtyRingState::collect`], keyed by KVM
    /// memslot id. This bridges a mid-round `KVM_EXIT_DIRTY_RING_FULL` harvest (done by
    /// a vCPU thread) to the next export (done by the VMM thread).
    accum: HashMap<u32, Vec<u64>>,
}

/// Drives the per-vCPU dirty rings for one VM. Shared (via `Arc`) between the VMM thread
/// (which exports) and the vCPU threads (which harvest on `KVM_EXIT_DIRTY_RING_FULL`).
#[derive(Debug)]
pub struct DirtyRingState {
    /// VM fd, used only for `KVM_RESET_DIRTY_RINGS`. Borrowed: the VM owns the fd and
    /// outlives this state (which lives in `VmCommon`).
    vm_fd: RawFd,
    /// Per-vCPU ring size in bytes.
    ring_bytes: usize,
    /// Whether `KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP` is active. When set, the caller must
    /// also harvest the per-memslot bitmap for pages dirtied outside vCPU context.
    pub with_bitmap: bool,
    inner: Mutex<DirtyRingInner>,
    /// Periodic-harvester control (see [`DirtyRingState::arm_harvester`]).
    harvester: Mutex<HarvesterCtl>,
}

/// Control block for the periodic background harvester.
#[derive(Debug)]
struct HarvesterCtl {
    /// A harvester thread is currently alive.
    running: bool,
    /// Drain/reset period.
    period: std::time::Duration,
    /// The thread parks itself for good once this passes (re-armed by each export).
    deadline: std::time::Instant,
}

impl DirtyRingState {
    /// Maps and registers the ring for a freshly-created vCPU. Must be called once per
    /// vCPU, after the vCPU fd exists and before it starts running.
    pub fn add_vcpu_ring(&self, vcpu_fd: RawFd) -> Result<(), DirtyRingError> {
        let ring = mmap_ring(vcpu_fd, self.ring_bytes)?;
        self.inner.lock().expect("Poisoned lock").rings.push(ring);
        Ok(())
    }

    /// Drains all rings into the accumulator and re-protects the harvested GFNs. Used by
    /// vCPU threads on `KVM_EXIT_DIRTY_RING_FULL` — it records before it resets, so no
    /// dirty page is lost even though the export hasn't run yet.
    pub fn harvest(&self) {
        let mut inner = self.inner.lock().expect("Poisoned lock");
        self.harvest_locked(&mut inner);
    }

    /// Drains all rings, re-protects, and returns the pages dirtied since the previous
    /// call (keyed by memslot id), clearing the accumulator. Used by the VMM thread to
    /// produce a dirty bitmap for export. Atomic w.r.t. concurrent vCPU `harvest()`s.
    pub fn collect(&self) -> HashMap<u32, Vec<u64>> {
        let mut inner = self.inner.lock().expect("Poisoned lock");
        self.harvest_locked(&mut inner);
        std::mem::take(&mut inner.accum)
    }

    /// Arm (or re-arm) the periodic background harvester: drain the rings +
    /// `KVM_RESET_DIRTY_RINGS` every `period_ms`, so each reset re-protects only a
    /// few-ms batch of GFNs — the per-round one-big-reset stall (mmu pass + remote
    /// TLB shootdown + the write-fault storm that follows) becomes a smooth,
    /// imperceptible tax. Harvested pages land in the same accumulator the export's
    /// `collect()` consumes, so correctness is unchanged. The thread re-arms its TTL
    /// on every call and parks for good ~30s after the last one (i.e. when the
    /// migration's rounds stop calling) — dirty tracking outside migrations keeps
    /// the cheap once-per-export behaviour.
    pub fn arm_harvester(self: &std::sync::Arc<Self>, period_ms: u64) {
        const TTL: std::time::Duration = std::time::Duration::from_secs(30);
        let mut h = self.harvester.lock().expect("Poisoned lock");
        h.period = std::time::Duration::from_millis(period_ms.clamp(20, 5000));
        h.deadline = std::time::Instant::now() + TTL;
        if h.running {
            return;
        }
        h.running = true;
        drop(h);
        let me = std::sync::Arc::clone(self);
        // Same thread-spawn syscall profile as the background dirty dump (clone
        // CLONE_THREAD + mmap MAP_STACK + set_robust_list), and the RESET ioctl is
        // already in the seccomp filter — no new syscalls.
        std::thread::spawn(move || {
            eprintln!("dirty-ring harvester: running");
            loop {
                let (period, expired) = {
                    let h = me.harvester.lock().expect("Poisoned lock");
                    (h.period, std::time::Instant::now() >= h.deadline)
                };
                if expired {
                    me.harvester.lock().expect("Poisoned lock").running = false;
                    eprintln!("dirty-ring harvester: idle TTL reached; stopped");
                    return;
                }
                std::thread::sleep(period);
                let mut inner = me.inner.lock().expect("Poisoned lock");
                me.harvest_locked(&mut inner);
            }
        });
    }

    fn harvest_locked(&self, inner: &mut DirtyRingInner) {
        let t0 = std::time::Instant::now();
        let mut harvested = false;
        // Split the borrow so we can drain rings into accum simultaneously.
        let DirtyRingInner { rings, accum } = inner;
        for ring in rings.iter_mut() {
            harvested |= ring.harvest_into(accum);
        }
        let harvest_us = t0.elapsed().as_micros();
        if harvested {
            self.reset_rings();
            // KVM_RESET_DIRTY_RINGS re-protects every harvested GFN: mmu work + a
            // remote TLB flush whose cost scales with the harvested batch. Timing it
            // sizes the per-round guest stall (and the win of harvesting in small
            // periodic doses instead of one per-round batch).
            eprintln!(
                "dirty-ring harvest: {}us + reset {}us",
                harvest_us,
                t0.elapsed().as_micros() - harvest_us
            );
        }
    }

    /// One `KVM_RESET_DIRTY_RINGS` re-protects every GFN we marked `RESET` across all
    /// rings, in bulk.
    fn reset_rings(&self) {
        // SAFETY: `vm_fd` is a valid VM fd; the request takes no argument. The musl
        // `ioctl` request type is `c_int` vs `c_ulong` on glibc, hence the `as _` cast.
        let ret = unsafe { libc::ioctl(self.vm_fd, KVM_RESET_DIRTY_RINGS as _) };
        if ret < 0 {
            // Re-protect failing means the next round may re-observe already-shipped
            // pages (wasteful) but never *miss* a page, so warn rather than abort.
            warn!(
                "KVM_RESET_DIRTY_RINGS failed: {}",
                std::io::Error::last_os_error()
            );
        }
    }
}

/// mmap one vCPU's dirty ring at the documented offset.
fn mmap_ring(vcpu_fd: RawFd, bytes: usize) -> Result<DirtyRing, DirtyRingError> {
    let offset = i64::from(KVM_DIRTY_LOG_PAGE_OFFSET) * host_page_size() as i64;
    // SAFETY: a shared mapping of `bytes` at the vCPU's dirty-ring offset; `bytes` is the
    // size we passed to the enable_cap and is page-aligned.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            vcpu_fd,
            offset,
        )
    };
    if std::ptr::eq(addr, libc::MAP_FAILED) {
        return Err(DirtyRingError::Mmap(std::io::Error::last_os_error()));
    }
    Ok(DirtyRing {
        base: addr.cast(),
        count: u32::try_from(bytes / std::mem::size_of::<kvm_dirty_gfn>()).unwrap(),
        mmap_len: bytes,
        fetch_index: 0,
    })
}

/// Largest power-of-two ring size, in bytes, that is `<= host_max` and `<=`
/// [`DESIRED_RING_BYTES`] and `>=` one page. Returns 0 if even one page does not fit
/// (in which case the ring is unusable and we fall back to the bitmap path).
fn clamp_ring_bytes(host_max: usize) -> usize {
    let page = host_page_size();
    let target = DESIRED_RING_BYTES.min(host_max);
    if target < page {
        return 0;
    }
    // round down to a power of two (also keeps the entry count a power of two, as KVM
    // requires, since the entry size 16 divides any power of two >= 16).
    let bytes = 1usize << (usize::BITS - 1 - target.leading_zeros());
    if bytes < page { 0 } else { bytes }
}

/// Negotiates dirty-ring support against the host. `ACQ_REL` is preferred; the legacy
/// `KVM_CAP_DIRTY_LOG_RING` is the fallback. `KVM_CHECK_EXTENSION` on the ring cap
/// returns the host's maximum ring size in bytes.
fn detect(kvm_fd: &KvmFd) -> Option<DirtyRingCaps> {
    let acq_rel = kvm_fd.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING_ACQ_REL));
    let (cap, host_max) = if acq_rel > 0 {
        (KVM_CAP_DIRTY_LOG_RING_ACQ_REL, acq_rel)
    } else {
        let legacy = kvm_fd.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING));
        if legacy > 0 {
            (KVM_CAP_DIRTY_LOG_RING, legacy)
        } else {
            return None;
        }
    };

    let ring_bytes = clamp_ring_bytes(host_max as usize);
    if ring_bytes == 0 {
        return None;
    }

    let with_bitmap = kvm_fd.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP)) > 0;
    Some(DirtyRingCaps {
        cap,
        ring_bytes,
        with_bitmap,
    })
}

/// Enables the dirty ring on a freshly-created VM, if the host supports it. Must be
/// called before any vCPU or memslot is created. Returns `None` (and leaves the VM on
/// the classic bitmap path) when the ring is unavailable or cannot be enabled, so hosts
/// without ring support never regress.
pub fn enable(vm_fd: &VmFd, kvm_fd: &KvmFd) -> Option<DirtyRingState> {
    let caps = detect(kvm_fd)?;

    let mut cap = kvm_enable_cap {
        cap: caps.cap,
        ..Default::default()
    };
    cap.args[0] = caps.ring_bytes as u64;
    if let Err(err) = vm_fd.enable_cap(&cap) {
        // Reported as supported but enabling failed: fall back to the bitmap path. This
        // is safe because the ring is not active, so KVM keeps using the memslot bitmap.
        warn!("Failed to enable KVM dirty ring (cap {}): {err}; using bitmap", caps.cap);
        return None;
    }

    // WITH_BITMAP requires the ring to be enabled first (done above) and must precede any
    // memslot. If it fails to enable we keep the ring but record that the supplementary
    // bitmap is unavailable.
    let mut with_bitmap = caps.with_bitmap;
    if with_bitmap {
        let bcap = kvm_enable_cap {
            cap: KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP,
            ..Default::default()
        };
        if let Err(err) = vm_fd.enable_cap(&bcap) {
            warn!("Failed to enable KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP: {err}");
            with_bitmap = false;
        }
    }

    info!(
        "KVM dirty ring enabled: {} bytes/vCPU ({} entries), with_bitmap={}",
        caps.ring_bytes,
        caps.ring_bytes / std::mem::size_of::<kvm_dirty_gfn>(),
        with_bitmap
    );

    Some(DirtyRingState {
        vm_fd: vm_fd.as_raw_fd(),
        ring_bytes: caps.ring_bytes,
        with_bitmap,
        inner: Mutex::new(DirtyRingInner {
            rings: Vec::new(),
            accum: HashMap::new(),
        }),
        harvester: Mutex::new(HarvesterCtl {
            running: false,
            period: std::time::Duration::from_millis(150),
            deadline: std::time::Instant::now(),
        }),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use kvm_bindings::{KVM_MEM_LOG_DIRTY_PAGES, kvm_userspace_memory_region};
    use kvm_ioctls::{Kvm as KvmFd, VcpuExit};

    use super::*;

    // A self-contained KVM harness that exercises the exact production harvest path
    // (`DirtyRing::harvest_into`) against a real guest that dirties known pages. Mirrors
    // the kvm-ioctls dirty-log doctest, but drives the ring instead of the bitmap.
    //
    // Real-mode payload, loaded at guest 0x1000. It writes a byte into each of the next
    // few pages (0x2000, 0x3000, 0x4000) then forces an MMIO exit so the harness can
    // harvest deterministically.
    #[cfg(target_arch = "x86_64")]
    fn run_ring_dirty_test(ring_bytes: usize) {
        let page_size = host_page_size();
        let mem_size = 0x10000;
        let guest_addr: u64 = 0x1000;

        let kvm = KvmFd::new().unwrap();
        if detect(&kvm).is_none() {
            // Host without ring support: nothing to test.
            return;
        }

        let vm = kvm.create_vm().unwrap();

        // Enable the ring BEFORE creating the vCPU, exactly as production does.
        let mut cap = kvm_enable_cap {
            cap: KVM_CAP_DIRTY_LOG_RING_ACQ_REL,
            ..Default::default()
        };
        cap.args[0] = ring_bytes as u64;
        // Prefer ACQ_REL; fall back to legacy if needed.
        if vm.enable_cap(&cap).is_err() {
            cap.cap = KVM_CAP_DIRTY_LOG_RING;
            vm.enable_cap(&cap).unwrap();
        }

        // Map guest memory with dirty logging on.
        let load_addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mem_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_ANONYMOUS | libc::MAP_SHARED | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(load_addr, libc::MAP_FAILED);
        let load_addr = load_addr.cast::<u8>();

        let mem_region = kvm_userspace_memory_region {
            slot: 0,
            guest_phys_addr: 0,
            memory_size: mem_size as u64,
            userspace_addr: load_addr as u64,
            flags: KVM_MEM_LOG_DIRTY_PAGES,
        };
        unsafe { vm.set_user_memory_region(mem_region).unwrap() };

        // mov al, 1
        // mov [0x2000], al
        // mov [0x3000], al
        // mov [0x4000], al
        // out 0x10, al        ; force a PIO/MMIO-style exit
        // hlt
        #[rustfmt::skip]
        let asm_code: &[u8] = &[
            0xb0, 0x01,                         // mov al, 1
            0xa2, 0x00, 0x20, 0x00, 0x00,       // mov [0x2000], al
            0xa2, 0x00, 0x30, 0x00, 0x00,       // mov [0x3000], al
            0xa2, 0x00, 0x40, 0x00, 0x00,       // mov [0x4000], al
            0xe6, 0x10,                         // out 0x10, al
            0xf4,                               // hlt
        ];
        unsafe {
            std::ptr::copy_nonoverlapping(asm_code.as_ptr(), load_addr.add(guest_addr as usize), asm_code.len());
        }

        let mut vcpu = vm.create_vcpu(0).unwrap();
        let mut sregs = vcpu.get_sregs().unwrap();
        sregs.cs.base = 0;
        sregs.cs.selector = 0;
        vcpu.set_sregs(&sregs).unwrap();
        let mut regs = vcpu.get_regs().unwrap();
        regs.rip = guest_addr;
        regs.rflags = 2;
        vcpu.set_regs(&regs).unwrap();

        // Map this vCPU's ring on its own fd, as production does post-creation.
        let mut ring = mmap_ring(vcpu.as_raw_fd(), ring_bytes).unwrap();

        // Run until the guest forces its exit.
        loop {
            match vcpu.run().unwrap() {
                VcpuExit::IoOut(0x10, _) => break,
                VcpuExit::Hlt => break,
                VcpuExit::MmioWrite(..) => {}
                other => panic!("unexpected exit: {other:?}"),
            }
        }

        // Harvest via the ring.
        let mut accum: HashMap<u32, Vec<u64>> = HashMap::new();
        let harvested = ring.harvest_into(&mut accum);
        assert!(harvested, "expected the guest's writes to be logged in the ring");

        // Re-protect the harvested GFNs (one VM-wide ioctl).
        let ret = unsafe { libc::ioctl(vm.as_raw_fd(), KVM_RESET_DIRTY_RINGS as _) };
        assert!(ret >= 0, "KVM_RESET_DIRTY_RINGS failed: {}", std::io::Error::last_os_error());

        // Collect the dirtied page indices for slot 0.
        let bitmap = accum.get(&0).expect("slot 0 should have dirty pages");
        let mut dirty_pages: Vec<u64> = Vec::new();
        for (i, word) in bitmap.iter().enumerate() {
            for b in 0..64 {
                if word & (1 << b) != 0 {
                    dirty_pages.push((i as u64) * 64 + b);
                }
            }
        }

        // The code page (0x1000) plus the three written pages (0x2000/0x3000/0x4000)
        // must all be present. (The code page is dirtied by the in-guest write path /
        // initial fault; the three data pages by the explicit stores.)
        let code_page = guest_addr / page_size as u64;
        for expected in [code_page, 0x2000 / page_size as u64, 0x3000 / page_size as u64, 0x4000 / page_size as u64] {
            assert!(
                dirty_pages.contains(&expected),
                "page {expected:#x} missing from harvested set {dirty_pages:?}"
            );
        }

        // A second harvest with no further writes must yield nothing (the reset moved the
        // ring's reset index forward; no entries are dirty).
        let mut accum2 = HashMap::new();
        assert!(!ring.harvest_into(&mut accum2), "no new writes => empty harvest");
        assert!(accum2.is_empty());

        unsafe { libc::munmap(load_addr.cast(), mem_size) };
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_ring_harvest_matches_writes() {
        // 64 KiB ring (4096 entries) is plenty for this tiny test and is a valid host
        // size on any ring-capable kernel.
        run_ring_dirty_test(64 * 1024);
    }

    #[test]
    fn test_clamp_ring_bytes() {
        let page = host_page_size();
        // Host max above desired -> clamp to desired (a power of two already).
        assert_eq!(clamp_ring_bytes(DESIRED_RING_BYTES * 4), DESIRED_RING_BYTES);
        // Host max equal to desired.
        assert_eq!(clamp_ring_bytes(DESIRED_RING_BYTES), DESIRED_RING_BYTES);
        // Non-power-of-two host max rounds down.
        assert_eq!(clamp_ring_bytes(DESIRED_RING_BYTES + 1), DESIRED_RING_BYTES);
        assert_eq!(clamp_ring_bytes(3 * page), 2 * page);
        // Exactly one page is usable.
        assert_eq!(clamp_ring_bytes(page), page);
        // Below a page is not.
        assert_eq!(clamp_ring_bytes(page - 1), 0);
        assert_eq!(clamp_ring_bytes(0), 0);
        // Every result is a power of two.
        for hm in [page, page * 5, DESIRED_RING_BYTES, DESIRED_RING_BYTES * 7 + 3] {
            let b = clamp_ring_bytes(hm);
            assert!(b == 0 || b.is_power_of_two(), "{b} not power of two");
        }
    }

    // Diagnostic (run explicitly: `cargo test ... -- --ignored --nocapture`). Prints the
    // negotiated ring parameters for the current host so we can confirm capacity and
    // WITH_BITMAP availability used by the performance analysis.
    #[test]
    #[ignore]
    fn diag_print_detection() {
        let kvm = KvmFd::new().unwrap();
        let page = host_page_size();
        let acq_rel = kvm.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING_ACQ_REL));
        let legacy = kvm.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING));
        let with_bitmap = kvm.check_extension_raw(u64::from(KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP));
        println!("host_page_size            = {page}");
        println!("CAP_DIRTY_LOG_RING_ACQ_REL= {acq_rel} (host max ring bytes)");
        println!("CAP_DIRTY_LOG_RING        = {legacy}");
        println!("CAP_..._WITH_BITMAP       = {with_bitmap}");
        match detect(&kvm) {
            Some(c) => println!(
                "detect() => cap={} ring_bytes={} ({} entries) with_bitmap={}",
                c.cap,
                c.ring_bytes,
                c.ring_bytes / std::mem::size_of::<kvm_dirty_gfn>(),
                c.with_bitmap
            ),
            None => println!("detect() => None (no ring support)"),
        }
    }

    #[test]
    fn test_detect_runs() {
        // Smoke: detection must not panic and must agree with itself.
        let kvm = KvmFd::new().unwrap();
        let a = detect(&kvm).is_some();
        let b = detect(&kvm).is_some();
        assert_eq!(a, b);
    }

    // ---------------------------------------------------------------------------------
    // Performance benchmark: per-round guest stall, dirty ring vs. classic bitmap.
    //
    // Boots a long-mode guest that endlessly bumps a host-visible heartbeat counter and
    // dirties a ~192 MiB region (one byte per 4 KiB page). A separate harvester thread
    // collects the dirty set on a cadence; we measure how long the guest's heartbeat
    // freezes around each harvest (the "per-round guest stall"). The classic path
    // (`KVM_GET_DIRTY_LOG`) re-write-protects the whole dirtied set synchronously,
    // stalling the vCPU; the ring path (`KVM_RESET_DIRTY_RINGS`) re-protects only the
    // harvested GFNs in bulk.
    //
    // Run explicitly:
    //   cargo test -p vmm --release --lib vstate::dirty_ring::tests::bench_per_round_stall \
    //       -- --ignored --nocapture
    // ---------------------------------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    mod bench {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        use kvm_bindings::{KVM_MEM_LOG_DIRTY_PAGES, kvm_userspace_memory_region};
        use kvm_ioctls::{Kvm as KvmFd, VcpuExit};

        use super::super::*;

        const NVCPU: u64 = 4;
        const GUEST_RAM: usize = 512 << 20; // 512 MiB
        const HEARTBEAT_GPA: u64 = 0x8000; // per-vCPU max-stall slot i at +8*i
        const PML4: u64 = 0x9000;
        const PDPTE: u64 = 0xa000;
        const PDE: u64 = 0xb000;
        const CODE_GPA: u64 = 0x10_0000; // 1 MiB
        const DIRTY_BASE: u64 = 0x20_0000; // 2 MiB, start of vCPU 0's region
        const PAGES_PER_VCPU: u64 = 24576; // 96 MiB / vCPU -> 384 MiB total dirtied
        const REGION_STRIDE: u64 = PAGES_PER_VCPU * 0x1000; // disjoint per-vCPU regions

        /// Outcome of one benchmark run.
        #[derive(Default, Clone, Copy)]
        struct RunStats {
            /// Worst guest-observed iteration gap with NO harvesting (warm-up floor).
            quiet_max_cycles: u64,
            /// Worst guest-observed iteration gap WHILE harvesting on a cadence.
            harvest_max_cycles: u64,
            /// Average dirty pages collected per harvest round.
            avg_pages: f64,
            /// Average harvest-call duration (microseconds).
            avg_harvest_us: u128,
        }

        /// Calibrate the host TSC frequency (cycles/sec) so guest cycle deltas convert to
        /// microseconds.
        fn tsc_hz() -> f64 {
            let t0 = Instant::now();
            let c0 = unsafe { core::arch::x86_64::_rdtsc() };
            std::thread::sleep(Duration::from_millis(200));
            let c1 = unsafe { core::arch::x86_64::_rdtsc() };
            (c1 - c0) as f64 / t0.elapsed().as_secs_f64()
        }

        /// Assembles the tiny 64-bit payload with the system assembler and returns raw
        /// machine code. Returns `None` if `as`/`objcopy` are unavailable.
        fn assemble_payload() -> Option<Vec<u8>> {
            use std::io::Write;
            let dir = std::env::temp_dir().join(format!("fc_dr_bench_{}", std::process::id()));
            std::fs::create_dir_all(&dir).ok()?;
            let asm = dir.join("p.s");
            let obj = dir.join("p.o");
            let bin = dir.join("p.bin");
            // Each vCPU walks its (disjoint) dirty region writing one byte per page, and
            // on every iteration measures the TSC delta since the previous iteration. It
            // keeps the running maximum delta in r10 and publishes it to its heartbeat
            // slot. The host reads those to learn the worst stall any vCPU experienced —
            // exactly the per-round freeze a re-protect + TLB-shootdown storm causes.
            //
            // Inputs (set per-vCPU via initial registers): rsi = heartbeat slot,
            // rbx = region base, r8 = pages in the region.
            let src = r#".intel_syntax noprefix
.code64
.global _start
_start:
    xor r10, r10           # r10 = running max delta (cycles)
    rdtsc
    shl rdx, 32
    or  rax, rdx
    mov r9, rax            # r9  = last tsc
restart:
    mov rdi, rbx           # rdi = current page pointer
    mov rcx, r8            # rcx = page counter
pg:
    rdtsc
    shl rdx, 32
    or  rax, rdx           # rax = now
    mov r11, rax
    sub rax, r9            # rax = now - last
    mov r9, r11            # last = now
    cmp rax, r10
    jbe skip
    mov r10, rax           # new max
    mov [rsi], r10         # publish it
skip:
    mov byte ptr [rdi], r10b   # dirty this page
    add rdi, 0x1000
    dec rcx
    jnz pg
    jmp restart
"#;
            std::fs::File::create(&asm).ok()?.write_all(src.as_bytes()).ok()?;
            let ok = std::process::Command::new("as")
                .args(["--64", "-o"])
                .arg(&obj)
                .arg(&asm)
                .status()
                .ok()?
                .success();
            if !ok {
                return None;
            }
            let ok = std::process::Command::new("objcopy")
                .args(["-O", "binary"])
                .arg(&obj)
                .arg(&bin)
                .status()
                .ok()?
                .success();
            if !ok {
                return None;
            }
            std::fs::read(&bin).ok()
        }

        /// Sets identity 2 MiB page tables covering [0, 1 GiB) plus a minimal GDT.
        unsafe fn write_paging(mem: *mut u8) {
            let w64 = |gpa: u64, val: u64| unsafe {
                std::ptr::write(mem.add(gpa as usize).cast::<u64>(), val)
            };
            w64(PML4, PDPTE | 0x03);
            w64(PDPTE, PDE | 0x03);
            for i in 0..512u64 {
                w64(PDE + i * 8, (i << 21) | 0x83);
            }
            // GDT @ 0x500: null, 64-bit code, data.
            w64(0x500, 0);
            w64(0x508, 0x00AF_9B00_0000_FFFF);
            w64(0x510, 0x00CF_9300_0000_FFFF);
        }

        extern "C" fn noop_handler(_: libc::c_int) {}

        fn bench_one(use_ring: bool, code: &[u8]) -> RunStats {
            let kvm = KvmFd::new().unwrap();
            let vm = kvm.create_vm().unwrap();

            let ring_bytes = if use_ring {
                let caps = detect(&kvm).expect("ring support required for ring benchmark");
                let mut cap = kvm_enable_cap {
                    cap: caps.cap,
                    ..Default::default()
                };
                cap.args[0] = caps.ring_bytes as u64;
                vm.enable_cap(&cap).unwrap();
                Some(caps.ring_bytes)
            } else {
                None
            };

            // Guest RAM (shared so the host can read the heartbeat).
            let mem = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    GUEST_RAM,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANONYMOUS | libc::MAP_SHARED | libc::MAP_NORESERVE,
                    -1,
                    0,
                )
            };
            assert_ne!(mem, libc::MAP_FAILED);
            let mem = mem.cast::<u8>();
            unsafe {
                write_paging(mem);
                std::ptr::copy_nonoverlapping(code.as_ptr(), mem.add(CODE_GPA as usize), code.len());
            }

            let region = kvm_userspace_memory_region {
                slot: 0,
                guest_phys_addr: 0,
                memory_size: GUEST_RAM as u64,
                userspace_addr: mem as u64,
                flags: KVM_MEM_LOG_DIRTY_PAGES,
            };
            unsafe { vm.set_user_memory_region(region).unwrap() };

            // Shared long-mode sregs (all vCPUs use the same identity page tables / GDT).
            let seg = |selector, type_, l, db| kvm_bindings::kvm_segment {
                base: 0,
                limit: 0xfffff,
                selector,
                type_,
                present: 1,
                dpl: 0,
                db,
                s: 1,
                l,
                g: 1,
                avl: 0,
                ..Default::default()
            };

            // Install a no-op SIGUSR1 handler so we can interrupt KVM_RUN to stop.
            unsafe {
                libc::signal(libc::SIGUSR1, noop_handler as *const () as libc::sighandler_t);
            }

            let stop = Arc::new(AtomicBool::new(false));
            let vm_rawfd = vm.as_raw_fd();

            // Per-vCPU: create it, configure long mode + its region registers, map its
            // ring, and launch a thread that runs it until interrupted.
            let mut rings: Vec<DirtyRing> = Vec::new();
            let mut tids: Vec<Arc<AtomicU64>> = Vec::new();
            let mut threads = Vec::new();
            for i in 0..NVCPU {
                let mut vcpu = vm.create_vcpu(i as u64).unwrap();
                let mut sregs = vcpu.get_sregs().unwrap();
                sregs.cr3 = PML4;
                sregs.cr4 = 0x20; // PAE
                sregs.cr0 = 0x8000_0001 | 0x10; // PG | PE | ET
                sregs.efer = 0x500; // LME | LMA
                sregs.gdt.base = 0x500;
                sregs.gdt.limit = 0x17;
                sregs.cs = seg(0x08, 0xb, 1, 0);
                let ds = seg(0x10, 0x3, 0, 1);
                sregs.ds = ds;
                sregs.es = ds;
                sregs.fs = ds;
                sregs.gs = ds;
                sregs.ss = ds;
                vcpu.set_sregs(&sregs).unwrap();
                let mut regs = vcpu.get_regs().unwrap();
                regs.rip = CODE_GPA;
                regs.rflags = 2;
                regs.rsi = HEARTBEAT_GPA + 8 * i; // this vCPU's max-stall slot
                regs.rbx = DIRTY_BASE + i * REGION_STRIDE; // this vCPU's region
                regs.r8 = PAGES_PER_VCPU;
                vcpu.set_regs(&regs).unwrap();

                if let Some(b) = ring_bytes {
                    rings.push(mmap_ring(vcpu.as_raw_fd(), b).unwrap());
                }

                let stop = stop.clone();
                let tid = Arc::new(AtomicU64::new(0));
                tids.push(tid.clone());
                threads.push(std::thread::spawn(move || {
                    tid.store(unsafe { libc::pthread_self() } as u64, Ordering::SeqCst);
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        match vcpu.run() {
                            // Dirty ring full: drain+reset so the guest can proceed.
                            // (Should not happen here: per-vCPU region < ring capacity.)
                            Ok(VcpuExit::Unsupported(31)) => {
                                unsafe { libc::ioctl(vm_rawfd, KVM_RESET_DIRTY_RINGS as _) };
                            }
                            Ok(_) => {}
                            Err(e) if e.errno() == libc::EINTR => {}
                            Err(e) => panic!("vcpu run error: {e}"),
                        }
                    }
                }));
            }

            // Worst stall across all vCPUs (each publishes its running max to its slot).
            let read_max = || -> u64 {
                (0..NVCPU)
                    .map(|i| unsafe {
                        std::ptr::read_volatile(
                            mem.add((HEARTBEAT_GPA + 8 * i) as usize).cast::<u64>(),
                        )
                    })
                    .max()
                    .unwrap_or(0)
            };

            // Wait for all vCPUs to start publishing.
            let t_start = Instant::now();
            while read_max() == 0 && t_start.elapsed() < Duration::from_secs(5) {
                std::hint::spin_loop();
            }

            // Phase A: quiet warm-up with NO harvesting. Establishes the floor (cold
            // faults, scheduler jitter) the running-max would record anyway.
            std::thread::sleep(Duration::from_millis(1200));
            let quiet_max_cycles = read_max();

            // Phase B: harvest on a 40 ms cadence while all vCPUs keep dirtying. Any
            // re-protect / TLB-shootdown freeze grows the worst-vCPU running max.
            let mut pages_total: u64 = 0;
            let mut harvest_us_total: u128 = 0;
            const ROUNDS: u64 = 40;
            for _ in 0..ROUNDS {
                std::thread::sleep(Duration::from_millis(40));
                let t0 = Instant::now();
                let pages: u64 = if ring_bytes.is_some() {
                    let mut accum = std::collections::HashMap::new();
                    for ring in rings.iter_mut() {
                        ring.harvest_into(&mut accum);
                    }
                    let r = unsafe { libc::ioctl(vm_rawfd, KVM_RESET_DIRTY_RINGS as _) };
                    assert!(r >= 0);
                    accum
                        .values()
                        .map(|v| v.iter().map(|w| w.count_ones() as u64).sum::<u64>())
                        .sum()
                } else {
                    let bm = vm.get_dirty_log(0, GUEST_RAM).unwrap();
                    bm.iter().map(|w| w.count_ones() as u64).sum()
                };
                harvest_us_total += t0.elapsed().as_micros();
                pages_total += pages;
            }
            let harvest_max_cycles = read_max();

            let stats = RunStats {
                quiet_max_cycles,
                harvest_max_cycles,
                avg_pages: pages_total as f64 / ROUNDS as f64,
                avg_harvest_us: harvest_us_total / ROUNDS as u128,
            };

            // Stop all vCPU threads.
            stop.store(true, Ordering::SeqCst);
            for tid in &tids {
                let t = tid.load(Ordering::SeqCst);
                if t != 0 {
                    unsafe { libc::pthread_kill(t as libc::pthread_t, libc::SIGUSR1) };
                }
            }
            for t in threads {
                t.join().unwrap();
            }

            drop(rings);
            unsafe { libc::munmap(mem.cast(), GUEST_RAM) };
            stats
        }

        fn report(name: &str, s: &RunStats, hz: f64) -> f64 {
            let to_us = |c: u64| (c as f64) / hz * 1e6;
            let quiet = to_us(s.quiet_max_cycles);
            let harvest = to_us(s.harvest_max_cycles);
            println!(
                "[{name}] avg_pages={:.0} (~{:.0} MiB/round) avg_harvest_call={}us | \
                 worst guest stall: quiet={:.0}us  with-harvest={:.0}us",
                s.avg_pages,
                s.avg_pages * 4096.0 / (1024.0 * 1024.0),
                s.avg_harvest_us,
                quiet,
                harvest,
            );
            harvest
        }

        #[test]
        #[ignore]
        fn bench_per_round_stall() {
            let code = match assemble_payload() {
                Some(c) => c,
                None => {
                    println!("SKIP: could not assemble payload (need `as`/`objcopy`)");
                    return;
                }
            };
            if detect(&KvmFd::new().unwrap()).is_none() {
                println!("SKIP: host has no dirty-ring support");
                return;
            }

            let hz = tsc_hz();
            println!("calibrated TSC ~= {:.2} GHz", hz / 1e9);

            println!("=== dirty bitmap (classic KVM_GET_DIRTY_LOG, re-protects on read) ===");
            let bitmap = bench_one(false, &code);
            let bitmap_stall = report("bitmap", &bitmap, hz);

            println!("=== dirty ring (KVM_RESET_DIRTY_RINGS, targeted re-protect) ===");
            let ring = bench_one(true, &code);
            let ring_stall = report("ring", &ring, hz);

            println!(
                "=== worst per-round guest stall: ring {:.0}us  vs  bitmap {:.0}us ===",
                ring_stall, bitmap_stall
            );

            // Correctness: both mechanisms must harvest essentially the same dirty volume.
            let parity = (ring.avg_pages - bitmap.avg_pages).abs() / bitmap.avg_pages.max(1.0);
            assert!(
                parity < 0.10,
                "ring/bitmap harvested volume differ by {:.0}% (ring {:.0} vs bitmap {:.0})",
                parity * 100.0,
                ring.avg_pages,
                bitmap.avg_pages
            );

            // The ring's worst per-round guest stall must stay under the spec's 50 ms
            // target. NOTE: this synthetic single-process benchmark does NOT reproduce the
            // multi-second bitmap stall seen with the real neko workload (idle spare
            // cores, no device/fault pattern), so we do NOT gate on ring < bitmap here —
            // the real perf proof is the neko migration on the deploy nodes.
            assert!(
                ring_stall < 50_000.0,
                "ring worst guest stall {ring_stall:.0}us exceeds 50ms target"
            );
        }
    }
}
