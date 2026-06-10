// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configurations used in the snapshotting context.

use std::path::PathBuf;

/// For crates that depend on `vmm` we export.
pub use semver::Version;
use serde::{Deserialize, Serialize};

/// The snapshot type options that are available when
/// creating a new snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub enum SnapshotType {
    /// Diff snapshot.
    Diff,
    /// Full snapshot.
    #[default]
    Full,
}

/// Specifies the method through which guest memory will get populated when
/// resuming from a snapshot:
/// 1) A file that contains the guest memory to be loaded,
/// 2) An UDS where a custom page-fault handler process is listening for the UFFD set up by
///    Firecracker to handle its guest memory page faults.
#[derive(Debug, Default, PartialEq, Eq, Deserialize)]
pub enum MemBackendType {
    /// Guest memory contents will be loaded from a file.
    #[default]
    File,
    /// Guest memory will be served through UFFD by a separate process.
    Uffd,
    /// Hybrid: guest memory is a memfd seeded from a base mem file (kernel-served,
    /// no fault tax); only the not-yet-arrived dirty pages are registered with
    /// UFFD_MINOR and served by a separate process via UFFDIO_CONTINUE.
    Hybrid,
    /// Precopy: map a shared base (`base_mem_path`) MAP_PRIVATE (clean pages shared
    /// COW across VMs, lazily cached — works on an ext4 file, no shmem needed) and
    /// eagerly apply the dirty delta (`backend_path` = delta file, `dirty_pages_path`
    /// = offsets) into guest memory before resume. No UFFD.
    Precopy,
}

/// Stores the configuration that will be used for creating a snapshot.
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateSnapshotParams {
    /// This marks the type of snapshot we want to create.
    /// The default value is `Full`, which means a full snapshot.
    #[serde(default = "SnapshotType::default")]
    pub snapshot_type: SnapshotType,
    /// Path to the file that will contain the microVM state.
    pub snapshot_path: PathBuf,
    /// Path to the file that will contain the guest memory.
    pub mem_file_path: PathBuf,
}

/// Stores the configuration that will be used for creating a state-only snapshot.
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CreateSnapshotStateParams {
    /// Path to the file that will contain the microVM state.
    pub snapshot_path: PathBuf,
    /// Whether to skip durable storage sync for the state file.
    #[serde(default)]
    pub no_sync: bool,
}

/// Stores the configuration that will be used for exporting only dirty guest memory.
#[derive(Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DirtyMemoryParams {
    /// Path to the file that will receive dirty guest memory pages.
    pub mem_file_path: PathBuf,
    /// Whether to sync the exported dirty memory file to durable storage.
    #[serde(default = "default_dirty_memory_sync")]
    pub sync: bool,
    /// Whether to mark virtio queue pages dirty before exporting memory.
    #[serde(default)]
    pub mark_virtio_queues: bool,
    /// Whether to perform the page copy on a background thread (live precopy
    /// rounds): the request returns after the dirty-bitmap capture, and the
    /// export file appears atomically (rename) when the copy completes. On
    /// failure `<mem_file_path>.err` is written instead. Keeps the VMM event
    /// loop — and so the guest's virtio I/O — running during a multi-GB dump.
    #[serde(default)]
    pub background: bool,
    /// [dirty-ring only] Arm a periodic background harvester with this period (ms):
    /// the rings are drained + `KVM_RESET_DIRTY_RINGS`-reset every `harvest_ms` into
    /// the accumulator, so each reset re-protects only a few-ms batch of pages — a
    /// smooth tax instead of the one big per-round re-protect + TLB-shootdown stall.
    /// Re-armed by every export that carries it; self-stops ~30s after the last one
    /// (i.e. when the migration's rounds end). 0/absent = per-round harvest as before.
    #[serde(default)]
    pub harvest_ms: u64,
}

fn default_dirty_memory_sync() -> bool {
    true
}

/// Allows for changing the mapping between tap devices and host devices
/// during snapshot restore
#[derive(Debug, PartialEq, Eq, Deserialize)]
pub struct NetworkOverride {
    /// The index of the interface to modify
    pub iface_id: String,
    /// The new name of the interface to be assigned
    pub host_dev_name: String,
}

/// Allows for changing the host UDS of the vsock backend during snapshot restore
#[derive(Debug, PartialEq, Eq, Deserialize)]
pub struct VsockOverride {
    /// The path to the UDS that will be used for the vsock interface
    pub uds_path: String,
}

/// Stores the configuration that will be used for loading a snapshot.
#[derive(Debug, PartialEq, Eq)]
pub struct LoadSnapshotParams {
    /// Path to the file that contains the microVM state to be loaded.
    pub snapshot_path: PathBuf,
    /// Specifies guest memory backend configuration.
    pub mem_backend: MemBackendConfig,
    /// Whether KVM dirty page tracking should be enabled, to space optimization
    /// of differential snapshots.
    pub track_dirty_pages: bool,
    /// When set to true, the vm is also resumed if the snapshot load
    /// is successful.
    pub resume_vm: bool,
    /// The network devices to override on load.
    pub network_overrides: Vec<NetworkOverride>,
    /// When set, the vsock backend UDS path will be overridden
    pub vsock_override: Option<VsockOverride>,
    /// [x86_64 only] When set to true, passes `KVM_CLOCK_REALTIME` to `KVM_SET_CLOCK` on restore,
    /// advancing kvmclock by the wall-clock time elapsed since the snapshot was taken. When false
    /// (default), kvmclock resumes from where it was at snapshot time.
    pub clock_realtime: bool,
    /// [Hybrid only] Two-phase restore. `Prepare` (callable while the migration source still
    /// runs): map the shared base + anon-overlay + UFFD-register the supplied (cumulative)
    /// dirty set, then HOLD the prepared memory + uffd — no handler handshake, no device
    /// restore, no resume. `Resume` (at cutover): overlay/register only this request's
    /// (small, final-round) dirty delta on the held memory, handshake with the handler,
    /// restore devices/vCPUs and (optionally) resume. `None` = the one-shot load, unchanged.
    pub phase: Option<LoadPhase>,
}

/// Phase discriminator for the two-phase Hybrid restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoadPhase {
    /// Map base + overlay/register cumulative dirty; hold memory + uffd.
    Prepare,
    /// Finish on prepared memory: register the delta, handshake, restore, resume.
    Resume,
}

/// Stores the configuration for loading a snapshot that is provided by the user.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadSnapshotConfig {
    /// Path to the file that contains the microVM state to be loaded.
    pub snapshot_path: PathBuf,
    /// Path to the file that contains the guest memory to be loaded. To be used only if
    /// `mem_backend` is not specified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_file_path: Option<PathBuf>,
    /// Guest memory backend configuration. Is not to be used in conjunction with `mem_file_path`.
    /// None value is allowed only if `mem_file_path` is present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mem_backend: Option<MemBackendConfig>,
    /// Whether or not to enable KVM dirty page tracking.
    #[serde(default)]
    #[deprecated]
    pub enable_diff_snapshots: bool,
    /// Whether KVM dirty page tracking should be enabled.
    #[serde(default)]
    pub track_dirty_pages: bool,
    /// Whether or not to resume the vm post snapshot load.
    #[serde(default)]
    pub resume_vm: bool,
    /// The network devices to override on load.
    #[serde(default)]
    pub network_overrides: Vec<NetworkOverride>,
    /// Whether or not to override the vsock backend UDS path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vsock_override: Option<VsockOverride>,
    /// [x86_64 only] When set to true, passes `KVM_CLOCK_REALTIME` to `KVM_SET_CLOCK` on restore.
    #[serde(default)]
    pub clock_realtime: bool,
    /// [Hybrid only] Two-phase restore discriminator ("prepare" | "resume"); absent = one-shot.
    #[serde(default)]
    pub phase: Option<LoadPhase>,
}

/// Stores the configuration used for managing snapshot memory.
#[derive(Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemBackendConfig {
    /// Path to the backend used to handle the guest memory.
    pub backend_path: PathBuf,
    /// Specifies the guest memory backend type.
    pub backend_type: MemBackendType,
    /// [Hybrid only] Base mem file used to seed the memfd (kernel-served pages).
    #[serde(default)]
    pub base_mem_path: Option<PathBuf>,
    /// [Hybrid only] File of dirty guest page offsets (one u64 per line) to
    /// register with UFFD_MINOR; everything else is served from the seeded base.
    /// In the fused Hybrid path this is the RESIDUAL set (pages re-dirtied around
    /// cutover) — the only pages served lazily; the rest are eager-applied below.
    #[serde(default)]
    pub dirty_pages_path: Option<PathBuf>,
    /// [Hybrid only] Delta file whose pages are eager-applied (COW'd private) into
    /// guest memory BEFORE resume — the rounds-shipped, already-local pages. Used
    /// with `eager_pages_path`. No UFFD on these.
    #[serde(default)]
    pub eager_delta_path: Option<PathBuf>,
    /// [Hybrid only] Offsets (one u64 per line) to eager-apply from
    /// `eager_delta_path` = the trusted set (dirty − residual). Disjoint from
    /// `dirty_pages_path`.
    #[serde(default)]
    pub eager_pages_path: Option<PathBuf>,
    /// [Hybrid only] Working-set manifest (one u64 per line) of base pages to
    /// prefault (make resident) before resume, for faster first-touch. Pages also
    /// in `dirty_pages_path` (residual/UFFD) are skipped so they still fault.
    #[serde(default)]
    pub prefill_pages_path: Option<PathBuf>,
}

/// The microVM state options.
#[derive(Debug, Deserialize, Serialize)]
pub enum VmState {
    /// The microVM is paused, which means that we can create a snapshot of it.
    Paused,
    /// The microVM is resumed; this state should be set after we load a snapshot.
    Resumed,
}

/// Keeps the microVM state necessary in the snapshotting context.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Vm {
    /// The microVM state, which can be `paused` or `resumed`.
    pub state: VmState,
}
