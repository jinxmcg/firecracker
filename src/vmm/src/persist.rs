// Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! Defines state structures for saving/restoring a Firecracker microVM.

use std::fmt::Debug;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write};
use std::mem::forget;
use std::os::fd::FromRawFd;
use std::os::unix::io::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use semver::Version;
use serde::{Deserialize, Serialize};
use userfaultfd::{FeatureFlags, RegisterMode, Uffd, UffdBuilder};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

#[cfg(target_arch = "aarch64")]
use crate::arch::aarch64::vcpu::get_manufacturer_id_from_host;
use crate::builder::{self, BuildMicrovmFromSnapshotError};
use crate::cpu_config::templates::StaticCpuTemplate;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::CpuidTrait;
#[cfg(target_arch = "x86_64")]
use crate::cpu_config::x86_64::cpuid::common::get_vendor_id_from_host;
use crate::device_manager::{DevicePersistError, DevicesState};
use crate::logger::{info, warn};
use crate::resources::VmResources;
use crate::seccomp::BpfThreadMap;
use crate::snapshot::Snapshot;
use crate::utils::u64_to_usize;
use crate::vmm_config::boot_source::BootSourceConfig;
use crate::vmm_config::instance_info::InstanceInfo;
use crate::vmm_config::machine_config::{HugePageConfig, MachineConfigError, MachineConfigUpdate};
use crate::vmm_config::snapshot::{
    CreateSnapshotParams, CreateSnapshotStateParams, LoadSnapshotParams, MemBackendConfig,
    MemBackendType,
};
use crate::vstate::kvm::KvmState;
use crate::vstate::memory::{
    self, GuestMemoryState, GuestRegionMmap, GuestRegionType, MemoryError,
};
use crate::vstate::vcpu::{VcpuSendEventError, VcpuState};
use crate::vstate::vm::{VmError, VmState};
use crate::{EventManager, Vmm, vstate};

/// Holds information related to the VM that is not part of VmState.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub struct VmInfo {
    /// Guest memory size.
    pub mem_size_mib: u64,
    /// smt information
    pub smt: bool,
    /// CPU template type
    pub cpu_template: StaticCpuTemplate,
    /// Boot source information.
    pub boot_source: BootSourceConfig,
    /// Huge page configuration
    pub huge_pages: HugePageConfig,
}

impl From<&VmResources> for VmInfo {
    fn from(value: &VmResources) -> Self {
        Self {
            mem_size_mib: value.machine_config.mem_size_mib as u64,
            smt: value.machine_config.smt,
            cpu_template: StaticCpuTemplate::from(&value.machine_config.cpu_template),
            boot_source: value.boot_source.config.clone(),
            huge_pages: value.machine_config.huge_pages,
        }
    }
}

impl From<&Vmm> for VmInfo {
    fn from(value: &Vmm) -> Self {
        let machine_config = &value.machine_config;
        Self {
            mem_size_mib: machine_config.mem_size_mib as u64,
            smt: machine_config.smt,
            cpu_template: StaticCpuTemplate::from(&machine_config.cpu_template),
            boot_source: value.boot_source_config.clone(),
            huge_pages: machine_config.huge_pages,
        }
    }
}

/// Contains the necessary state for saving/restoring a microVM.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct MicrovmState {
    /// Miscellaneous VM info.
    pub vm_info: VmInfo,
    /// KVM KVM state.
    pub kvm_state: KvmState,
    /// VM KVM state.
    pub vm_state: VmState,
    /// Vcpu states.
    pub vcpu_states: Vec<VcpuState>,
    /// Device states.
    pub device_states: DevicesState,
}

/// This describes the mapping between Firecracker base virtual address and
/// offset in the buffer or file backend for a guest memory region. It is used
/// to tell an external process/thread where to populate the guest memory data
/// for this range.
///
/// E.g. Guest memory contents for a region of `size` bytes can be found in the
/// backend at `offset` bytes from the beginning, and should be copied/populated
/// into `base_host_address`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GuestRegionUffdMapping {
    /// Base host virtual address where the guest memory contents for this
    /// region should be copied/populated.
    pub base_host_virt_addr: u64,
    /// Region size.
    pub size: usize,
    /// Offset in the backend file/buffer where the region contents are.
    pub offset: u64,
    /// The configured page size for this memory region.
    pub page_size: usize,
    /// The configured page size **in bytes** for this memory region. The name is
    /// wrong but cannot be changed due to being API, so this field is deprecated,
    /// to be removed in 2.0.
    #[deprecated]
    pub page_size_kib: usize,
}

/// Errors related to saving and restoring Microvm state.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum MicrovmStateError {
    /// Operation not allowed: {0}
    NotAllowed(String),
    /// Cannot restore devices: {0}
    RestoreDevices(#[from] DevicePersistError),
    /// Cannot save Vcpu state: {0}
    SaveVcpuState(vstate::vcpu::VcpuError),
    /// Cannot save KvmVm state: {0}
    SaveVmState(vstate::vm::KvmVmError),
    /// Cannot signal Vcpu: {0}
    SignalVcpu(VcpuSendEventError),
    /// Vcpu is in unexpected state.
    UnexpectedVcpuResponse,
}

/// Errors associated with creating a snapshot.
#[rustfmt::skip]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum CreateSnapshotError {
    /// Cannot get dirty bitmap: {0}
    DirtyBitmap(#[from] VmError),
    /// Cannot write memory file: {0}
    Memory(#[from] MemoryError),
    /// Cannot perform {0} on the memory backing file: {1}
    MemoryBackingFile(&'static str, io::Error),
    /// Cannot save the microVM state: {0}
    MicrovmState(MicrovmStateError),
    /// Cannot serialize the microVM state: {0}
    SerializeMicrovmState(#[from] crate::snapshot::SnapshotError),
    /// Cannot perform {0} on the snapshot backing file: {1}
    SnapshotBackingFile(&'static str, io::Error),
}

/// Snapshot version
pub const SNAPSHOT_VERSION: Version = Version::new(10, 0, 0);

/// Creates a Microvm snapshot.
pub fn create_snapshot(
    vmm: &mut Vmm,
    vm_info: &VmInfo,
    params: &CreateSnapshotParams,
) -> Result<(), CreateSnapshotError> {
    let microvm_state = vmm
        .save_state(vm_info)
        .map_err(CreateSnapshotError::MicrovmState)?;

    snapshot_state_to_file(&microvm_state, &params.snapshot_path, false)?;

    let kvm_vm = vmm.vm.as_kvm().ok_or_else(|| {
        CreateSnapshotError::MicrovmState(MicrovmStateError::NotAllowed(
            "snapshot requires KVM".into(),
        ))
    })?;
    kvm_vm.snapshot_memory_to_file(&params.mem_file_path, params.snapshot_type)?;

    // We need to mark queues as dirty again for all activated devices. The reason we
    // do it here is that we don't mark pages as dirty during runtime
    // for queue objects.
    vmm.device_manager
        .mark_virtio_queue_memory_dirty(kvm_vm.guest_memory());

    Ok(())
}

/// Creates a Microvm state-only snapshot.
pub fn create_state_snapshot(
    vmm: &mut Vmm,
    vm_info: &VmInfo,
    params: &CreateSnapshotStateParams,
) -> Result<(), CreateSnapshotError> {
    let microvm_state = vmm
        .save_state(vm_info)
        .map_err(CreateSnapshotError::MicrovmState)?;

    snapshot_state_to_file(&microvm_state, &params.snapshot_path, params.no_sync)
}

fn snapshot_state_to_file(
    microvm_state: &MicrovmState,
    snapshot_path: &Path,
    no_sync: bool,
) -> Result<(), CreateSnapshotError> {
    use self::CreateSnapshotError::*;
    let mut snapshot_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(snapshot_path)
        .map_err(|err| SnapshotBackingFile("open", err))?;

    let snapshot = Snapshot::new(microvm_state);
    snapshot.save(&mut snapshot_file)?;
    snapshot_file
        .flush()
        .map_err(|err| SnapshotBackingFile("flush", err))?;
    if no_sync {
        return Ok(());
    }
    snapshot_file
        .sync_all()
        .map_err(|err| SnapshotBackingFile("sync_all", err))
}

/// Validates that snapshot CPU vendor matches the host CPU vendor.
///
/// # Errors
///
/// When:
/// - Failed to read host vendor.
/// - Failed to read snapshot vendor.
#[cfg(target_arch = "x86_64")]
pub fn validate_cpu_vendor(microvm_state: &MicrovmState) {
    let host_vendor_id = get_vendor_id_from_host();
    let snapshot_vendor_id = microvm_state.vcpu_states[0].cpuid.vendor_id();
    match (host_vendor_id, snapshot_vendor_id) {
        (Ok(host_id), Some(snapshot_id)) => {
            info!("Host CPU vendor ID: {host_id:?}");
            info!("Snapshot CPU vendor ID: {snapshot_id:?}");
            if host_id != snapshot_id {
                warn!("Host CPU vendor ID differs from the snapshotted one",);
            }
        }
        (Ok(host_id), None) => {
            info!("Host CPU vendor ID: {host_id:?}");
            warn!("Snapshot CPU vendor ID: couldn't get from the snapshot");
        }
        (Err(_), Some(snapshot_id)) => {
            warn!("Host CPU vendor ID: couldn't get from the host");
            info!("Snapshot CPU vendor ID: {snapshot_id:?}");
        }
        (Err(_), None) => {
            warn!("Host CPU vendor ID: couldn't get from the host");
            warn!("Snapshot CPU vendor ID: couldn't get from the snapshot");
        }
    }
}

/// Validate that Snapshot Manufacturer ID matches
/// the one from the Host
///
/// The manufacturer ID for the Snapshot is taken from each VCPU state.
/// # Errors
///
/// When:
/// - Failed to read host vendor.
/// - Failed to read snapshot vendor.
#[cfg(target_arch = "aarch64")]
pub fn validate_cpu_manufacturer_id(microvm_state: &MicrovmState) {
    let host_cpu_id = get_manufacturer_id_from_host();
    let snapshot_cpu_id = microvm_state.vcpu_states[0].regs.manifacturer_id();
    match (host_cpu_id, snapshot_cpu_id) {
        (Some(host_id), Some(snapshot_id)) => {
            info!("Host CPU manufacturer ID: {host_id:?}");
            info!("Snapshot CPU manufacturer ID: {snapshot_id:?}");
            if host_id != snapshot_id {
                warn!("Host CPU manufacturer ID differs from the snapshotted one",);
            }
        }
        (Some(host_id), None) => {
            info!("Host CPU manufacturer ID: {host_id:?}");
            warn!("Snapshot CPU manufacturer ID: couldn't get from the snapshot");
        }
        (None, Some(snapshot_id)) => {
            warn!("Host CPU manufacturer ID: couldn't get from the host");
            info!("Snapshot CPU manufacturer ID: {snapshot_id:?}");
        }
        (None, None) => {
            warn!("Host CPU manufacturer ID: couldn't get from the host");
            warn!("Snapshot CPU manufacturer ID: couldn't get from the snapshot");
        }
    }
}
/// Error type for [`snapshot_state_sanity_check`].
#[derive(Debug, thiserror::Error, displaydoc::Display, PartialEq, Eq)]
pub enum SnapShotStateSanityCheckError {
    /// No memory region defined.
    NoMemory,
    /// No DRAM memory region defined.
    NoDramMemory,
    /// DRAM memory has more than a single slot.
    DramMemoryTooManySlots,
    /// DRAM memory is unplugged.
    DramMemoryUnplugged,
}

/// Performs sanity checks against the state file and returns specific errors.
pub fn snapshot_state_sanity_check(
    microvm_state: &MicrovmState,
) -> Result<(), SnapShotStateSanityCheckError> {
    // Check that the snapshot contains at least 1 mem region, that at least one is Dram,
    // and that Dram region contains a single plugged slot.
    // Upper bound check will be done when creating guest memory by comparing against
    // KVM max supported value kvm_context.max_memslots().
    let regions = &microvm_state.vm_state.memory.regions;

    if regions.is_empty() {
        return Err(SnapShotStateSanityCheckError::NoMemory);
    }

    if !regions
        .iter()
        .any(|r| r.region_type == GuestRegionType::Dram)
    {
        return Err(SnapShotStateSanityCheckError::NoDramMemory);
    }

    for dram_region in regions
        .iter()
        .filter(|r| r.region_type == GuestRegionType::Dram)
    {
        if dram_region.plugged.len() != 1 {
            return Err(SnapShotStateSanityCheckError::DramMemoryTooManySlots);
        }

        if !dram_region.plugged[0] {
            return Err(SnapShotStateSanityCheckError::DramMemoryUnplugged);
        }
    }

    #[cfg(target_arch = "x86_64")]
    validate_cpu_vendor(microvm_state);
    #[cfg(target_arch = "aarch64")]
    validate_cpu_manufacturer_id(microvm_state);

    Ok(())
}

/// Error type for [`restore_from_snapshot`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum RestoreFromSnapshotError {
    /// Failed to get snapshot state from file: {0}
    File(#[from] SnapshotStateFromFileError),
    /// Invalid snapshot state: {0}
    Invalid(#[from] SnapShotStateSanityCheckError),
    /// Failed to load guest memory: {0}
    GuestMemory(#[from] RestoreFromSnapshotGuestMemoryError),
    /// Failed to build microVM from snapshot: {0}
    Build(#[from] BuildMicrovmFromSnapshotError),
}
/// Sub-Error type for [`restore_from_snapshot`] to contain either [`GuestMemoryFromFileError`] or
/// [`GuestMemoryFromUffdError`] within [`RestoreFromSnapshotError`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum RestoreFromSnapshotGuestMemoryError {
    /// Error creating guest memory from file: {0}
    File(#[from] GuestMemoryFromFileError),
    /// Error creating guest memory from uffd: {0}
    Uffd(#[from] GuestMemoryFromUffdError),
    /// Error creating guest memory from hybrid: {0}
    Hybrid(#[from] GuestMemoryFromHybridError),
    /// Error creating guest memory from precopy: {0}
    Precopy(GuestMemoryFromHybridError),
}

/// Phase-A entry for the two-phase Hybrid restore: parse the snapshot only for the
/// memory layout + huge-page config, then map+overlay+register the cumulative dirty
/// set and hold the result (see [`PreparedHybrid`]). The snapshot passed here may be
/// the BASE snapshot — only its region geometry is used, and the resume phase
/// sanity-checks that geometry against the final snapshot.
pub fn prepare_hybrid_from_params(
    params: &LoadSnapshotParams,
) -> Result<PreparedHybrid, RestoreFromSnapshotError> {
    if params.mem_backend.backend_type != MemBackendType::Hybrid {
        return Err(RestoreFromSnapshotError::GuestMemory(
            RestoreFromSnapshotGuestMemoryError::Hybrid(GuestMemoryFromHybridError::InvalidPhase),
        ));
    }
    let microvm_state = snapshot_state_from_file(&params.snapshot_path)?;
    prepare_hybrid(
        &params.mem_backend,
        &microvm_state.vm_state.memory,
        params.track_dirty_pages,
        microvm_state.vm_info.huge_pages,
    )
    .map_err(|e| RestoreFromSnapshotError::GuestMemory(RestoreFromSnapshotGuestMemoryError::Hybrid(e)))
}

/// Loads a Microvm snapshot producing a 'paused' Microvm.
pub fn restore_from_snapshot(
    instance_info: &InstanceInfo,
    event_manager: &mut EventManager,
    seccomp_filters: &BpfThreadMap,
    params: &LoadSnapshotParams,
    vm_resources: &mut VmResources,
    prepared: Option<PreparedHybrid>,
) -> Result<Arc<Mutex<Vmm>>, RestoreFromSnapshotError> {
    let mut microvm_state = snapshot_state_from_file(&params.snapshot_path)?;
    for entry in &params.network_overrides {
        microvm_state
            .device_states
            .mmio_state
            .net_devices
            .iter_mut()
            .map(|device| &mut device.device_state)
            .chain(
                microvm_state
                    .device_states
                    .pci_state
                    .net_devices
                    .iter_mut()
                    .map(|device| &mut device.device_state),
            )
            .find(|x| x.id == entry.iface_id)
            .map(|device_state| device_state.tap_if_name.clone_from(&entry.host_dev_name))
            .ok_or(SnapshotStateFromFileError::UnknownNetworkDevice)?;
    }

    if let Some(vsock_override) = &params.vsock_override {
        // There should only ever be at most one vsock device, therefore this
        // should correctly find it and modify the path if such a device exists.
        let device_state = microvm_state
            .device_states
            .mmio_state
            .vsock_device
            .as_mut()
            .map(|device| &mut device.device_state)
            .or_else(|| {
                microvm_state
                    .device_states
                    .pci_state
                    .vsock_device
                    .as_mut()
                    .map(|device| &mut device.device_state)
            })
            .ok_or(SnapshotStateFromFileError::UnknownVsockDevice)?;

        device_state
            .backend
            .uds_path
            .clone_from(&vsock_override.uds_path);
    }

    let track_dirty_pages = params.track_dirty_pages;

    let vcpu_count = microvm_state
        .vcpu_states
        .len()
        .try_into()
        .map_err(|_| MachineConfigError::InvalidVcpuCount)
        .map_err(BuildMicrovmFromSnapshotError::VmUpdateConfig)?;

    vm_resources
        .update_machine_config(&MachineConfigUpdate {
            vcpu_count: Some(vcpu_count),
            mem_size_mib: Some(u64_to_usize(microvm_state.vm_info.mem_size_mib)),
            smt: Some(microvm_state.vm_info.smt),
            cpu_template: Some(microvm_state.vm_info.cpu_template),
            track_dirty_pages: Some(track_dirty_pages),
            huge_pages: Some(microvm_state.vm_info.huge_pages),
            #[cfg(feature = "gdb")]
            gdb_socket_path: None,
        })
        .map_err(BuildMicrovmFromSnapshotError::VmUpdateConfig)?;

    // Some sanity checks before building the microvm.
    snapshot_state_sanity_check(&microvm_state)?;

    let mem_backend_path = &params.mem_backend.backend_path;
    let mem_state = &microvm_state.vm_state.memory;

    let t_mem = std::time::Instant::now(); // guest-memory phase vs device/vCPU restore split
    let (guest_memory, uffd) = match params.mem_backend.backend_type {
        MemBackendType::File => {
            if vm_resources.machine_config.huge_pages.is_hugetlbfs() {
                return Err(RestoreFromSnapshotGuestMemoryError::File(
                    GuestMemoryFromFileError::HugetlbfsSnapshot,
                )
                .into());
            }
            (
                guest_memory_from_file(mem_backend_path, mem_state, track_dirty_pages)
                    .map_err(RestoreFromSnapshotGuestMemoryError::File)?,
                None,
            )
        }
        MemBackendType::Uffd => guest_memory_from_uffd(
            mem_backend_path,
            mem_state,
            track_dirty_pages,
            vm_resources.machine_config.huge_pages,
        )
        .map_err(RestoreFromSnapshotGuestMemoryError::Uffd)?,
        MemBackendType::Hybrid => match prepared {
            // two-phase resume: overlay/register only the final delta on the memory
            // the prepare phase already set up, then the deferred handler handshake
            Some(p) => finish_prepared_hybrid(p, &params.mem_backend, mem_state)
                .map_err(RestoreFromSnapshotGuestMemoryError::Hybrid)?,
            None => guest_memory_from_hybrid(
                &params.mem_backend,
                mem_state,
                track_dirty_pages,
                vm_resources.machine_config.huge_pages,
            )
            .map_err(RestoreFromSnapshotGuestMemoryError::Hybrid)?,
        },
        MemBackendType::Precopy => guest_memory_from_precopy(
            &params.mem_backend,
            mem_state,
            track_dirty_pages,
        )
        .map_err(RestoreFromSnapshotGuestMemoryError::Precopy)?,
    };
    let mem_us = t_mem.elapsed().as_micros();
    let t_build = std::time::Instant::now();
    let r = builder::build_microvm_from_snapshot(
        instance_info,
        event_manager,
        microvm_state,
        guest_memory,
        uffd,
        seccomp_filters,
        vm_resources,
        params.clock_realtime,
    )
    .map_err(RestoreFromSnapshotError::Build);
    eprintln!(
        "restore timing: guest-memory={}us build(devices+vcpus)={}us",
        mem_us,
        t_build.elapsed().as_micros()
    );
    r
}

/// Error type for [`snapshot_state_from_file`]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum SnapshotStateFromFileError {
    /// Failed to open snapshot file: {0}
    Open(#[from] std::io::Error),
    /// Failed to load snapshot state from file: {0}
    Load(#[from] crate::snapshot::SnapshotError),
    /// Unknown Network Device.
    UnknownNetworkDevice,
    /// Unknown Vsock Device.
    UnknownVsockDevice,
}

fn snapshot_state_from_file(
    snapshot_path: &Path,
) -> Result<MicrovmState, SnapshotStateFromFileError> {
    let mut snapshot_reader = File::open(snapshot_path)?;
    let snapshot = Snapshot::load(&mut snapshot_reader)?;

    Ok(snapshot.data)
}

/// Error type for [`guest_memory_from_file`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum GuestMemoryFromFileError {
    /// Failed to load guest memory: {0}
    File(#[from] std::io::Error),
    /// Failed to restore guest memory: {0}
    Restore(#[from] MemoryError),
    /// Cannot restore hugetlbfs backed snapshot by mapping the memory file. Please use uffd.
    HugetlbfsSnapshot,
}

fn guest_memory_from_file(
    mem_file_path: &Path,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
) -> Result<Vec<GuestRegionMmap>, GuestMemoryFromFileError> {
    let mem_file = File::open(mem_file_path)?;
    let guest_mem = memory::snapshot_file(mem_file, mem_state.regions(), track_dirty_pages)?;
    Ok(guest_mem)
}

/// Error type for [`guest_memory_from_uffd`]
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum GuestMemoryFromUffdError {
    /// Failed to restore guest memory: {0}
    Restore(#[from] MemoryError),
    /// Failed to UFFD object: {0}
    Create(userfaultfd::Error),
    /// Failed to register memory address range with the userfaultfd object: {0}
    Register(userfaultfd::Error),
    /// Failed to connect to UDS Unix stream: {0}
    Connect(#[from] std::io::Error),
    /// Failed to sends file descriptor: {0}
    Send(#[from] vmm_sys_util::errno::Error),
}

fn guest_memory_from_uffd(
    mem_uds_path: &Path,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromUffdError> {
    let (guest_memory, backend_mappings) =
        create_guest_memory(mem_state, track_dirty_pages, huge_pages)?;

    let mut uffd_builder = UffdBuilder::new();

    // We only make use of this if balloon devices are present, but we can enable it unconditionally
    // because the only place the kernel checks this is in a hook from madvise, e.g. it doesn't
    // actively change the behavior of UFFD, only passively. Without balloon devices
    // we never call madvise anyway, so no need to put this into a conditional.
    uffd_builder.require_features(FeatureFlags::EVENT_REMOVE);

    let uffd = uffd_builder
        .close_on_exec(true)
        .non_blocking(true)
        .user_mode_only(false)
        .create()
        .map_err(GuestMemoryFromUffdError::Create)?;

    for mem_region in guest_memory.iter() {
        uffd.register(mem_region.as_ptr().cast(), mem_region.size() as _)
            .map_err(GuestMemoryFromUffdError::Register)?;
    }

    send_uffd_handshake(mem_uds_path, &backend_mappings, &uffd)?;

    Ok((guest_memory, Some(uffd)))
}

fn create_guest_memory(
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Vec<GuestRegionUffdMapping>), GuestMemoryFromUffdError> {
    let guest_memory = memory::anonymous(mem_state.regions(), track_dirty_pages, huge_pages)?;
    let mut backend_mappings = Vec::with_capacity(guest_memory.len());
    let mut offset = 0;
    for mem_region in guest_memory.iter() {
        #[allow(deprecated)]
        backend_mappings.push(GuestRegionUffdMapping {
            base_host_virt_addr: mem_region.as_ptr() as u64,
            size: mem_region.size(),
            offset,
            page_size: huge_pages.page_size(),
            page_size_kib: huge_pages.page_size(),
        });
        offset += mem_region.size() as u64;
    }

    Ok((guest_memory, backend_mappings))
}

fn send_uffd_handshake(
    mem_uds_path: &Path,
    backend_mappings: &[GuestRegionUffdMapping],
    uffd: &impl AsRawFd,
) -> Result<(), GuestMemoryFromUffdError> {
    // This is safe to unwrap() because we control the contents of the vector
    // (i.e GuestRegionUffdMapping entries).
    let backend_mappings = serde_json::to_string(backend_mappings).unwrap();

    let socket = UnixStream::connect(mem_uds_path)?;
    socket.send_with_fd(
        backend_mappings.as_bytes(),
        // In the happy case we can close the fd since the other process has it open and is
        // using it to serve us pages.
        //
        // The problem is that if other process crashes/exits, firecracker guest memory
        // will simply revert to anon-mem behavior which would lead to silent errors and
        // undefined behavior.
        //
        // To tackle this scenario, the page fault handler can notify Firecracker of any
        // crashes/exits. There is no need for Firecracker to explicitly send its process ID.
        // The external process can obtain Firecracker's PID by calling `getsockopt` with
        // `libc::SO_PEERCRED` option like so:
        //
        // let mut val = libc::ucred { pid: 0, gid: 0, uid: 0 };
        // let mut ucred_size: u32 = mem::size_of::<libc::ucred>() as u32;
        // libc::getsockopt(
        //      socket.as_raw_fd(),
        //      libc::SOL_SOCKET,
        //      libc::SO_PEERCRED,
        //      &mut val as *mut _ as *mut _,
        //      &mut ucred_size as *mut libc::socklen_t,
        // );
        //
        // Per this linux man page: https://man7.org/linux/man-pages/man7/unix.7.html,
        // `SO_PEERCRED` returns the credentials (PID, UID and GID) of the peer process
        // connected to this socket. The returned credentials are those that were in effect
        // at the time of the `connect` call.
        //
        // Moreover, Firecracker holds a copy of the UFFD fd as well, so that even if the
        // page fault handler process does not tear down Firecracker when necessary, the
        // uffd will still be alive but with no one to serve faults, leading to guest freeze.
        uffd.as_raw_fd(),
    )?;

    // We prevent Rust from closing the socket file descriptor to avoid a potential race condition
    // between the mappings message and the connection shutdown. If the latter arrives at the UFFD
    // handler first, the handler never sees the mappings.
    forget(socket);

    Ok(())
}

// ---------- Hybrid backend (memfd base + UFFD_MINOR for the dirty tail) ----------

// Raw userfaultfd uapi: the safe `UffdBuilder` cannot negotiate MINOR_SHMEM
// (the crate's FeatureFlags don't expose it), so we do the UFFDIO_API by hand.
const UFFD_API: u64 = 0xAA;
const UFFDIO_API: libc::c_ulong = 0xC018_AA3F;
const UFFD_FEATURE_EVENT_REMOVE: u64 = 1 << 3;
const UFFD_FEATURE_MINOR_SHMEM: u64 = 1 << 10;

#[repr(C)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

/// Error type for [`guest_memory_from_hybrid`].
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum GuestMemoryFromHybridError {
    /// Hybrid backend requires `base_mem_path`
    MissingBase,
    /// Failed to restore guest memory: {0}
    Restore(#[from] MemoryError),
    /// Hybrid memfd/base I/O error: {0}
    Io(#[from] std::io::Error),
    /// Failed to create UFFD: {0}
    Uffd(std::io::Error),
    /// Failed to register UFFD_MINOR range: {0}
    Register(userfaultfd::Error),
    /// Failed to send fds to handler: {0}
    Send(vmm_sys_util::errno::Error),
    /// phase=resume without a prior phase=prepare (or eager/prefill fields in a two-phase request)
    InvalidPhase,
    /// prepared memory layout does not match this snapshot's regions
    PreparedMismatch,
}

/// Guest memory prepared by the two-phase Hybrid restore's `prepare` phase: the shared
/// base is mapped, the CUMULATIVE dirty set is anon-overlaid + UFFD-registered, and the
/// uffd is held UNATTACHED (no handler handshake yet). Nothing can fault on it — no
/// vCPUs run and no device state has been restored — so holding it across API calls is
/// safe. The `resume` phase overlays/registers only the final-round delta, performs the
/// handshake, and hands `guest_memory`+`uffd` to the normal microvm build.
#[derive(Debug)]
pub struct PreparedHybrid {
    guest_memory: Vec<GuestRegionMmap>,
    uffd: Uffd,
    backend_mappings: Vec<GuestRegionUffdMapping>,
    /// keeps the shared-base fd alive for the deferred handshake
    base: File,
    page_size: usize,
    /// (offset, size) of each region — sanity-checked against the resume snapshot
    region_shape: Vec<(u64, usize)>,
}

/// Phase A of the two-phase Hybrid restore: map the shared base, anon-overlay +
/// UFFD-register the supplied (cumulative) dirty set, and hold the result. Runs while
/// the migration source is still live — this is exactly the set-size-proportional work
/// that used to sit inside the resume blackout. Eager/prefill fields are not supported
/// in two-phase mode (an eager apply into registered pages would deadlock on the UFFD).
pub fn prepare_hybrid(
    mem_backend: &MemBackendConfig,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<PreparedHybrid, GuestMemoryFromHybridError> {
    if mem_backend.eager_delta_path.is_some()
        || mem_backend.eager_pages_path.is_some()
        || mem_backend.prefill_pages_path.is_some()
    {
        return Err(GuestMemoryFromHybridError::InvalidPhase);
    }
    let base_path = mem_backend
        .base_mem_path
        .as_ref()
        .ok_or(GuestMemoryFromHybridError::MissingBase)?;
    let t0 = std::time::Instant::now();
    let regions: Vec<_> = mem_state.regions().collect();
    let region_shape: Vec<(u64, usize)> = regions.iter().map(|r| (r.0 .0, r.1)).collect();
    let base = File::open(base_path)?;
    let base_clone = base.try_clone()?;
    let guest_memory = memory::snapshot_file(base, regions.iter().copied(), track_dirty_pages)?;

    let page_size = huge_pages.page_size();
    let mut backend_mappings = Vec::with_capacity(guest_memory.len());
    let mut offset = 0u64;
    for region in guest_memory.iter() {
        #[allow(deprecated)]
        backend_mappings.push(GuestRegionUffdMapping {
            base_host_virt_addr: region.as_ptr() as u64,
            size: region.size(),
            offset,
            page_size,
            page_size_kib: page_size,
        });
        offset += region.size() as u64;
    }

    let mut n = 0u64;
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        n = overlay_anon_ranges(&backend_mappings, dirty_path, page_size)?;
    }
    let uffd = create_missing_uffd()?;
    let mut reg = 0u64;
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        reg = register_minor_ranges(&uffd, &backend_mappings, dirty_path, page_size, RegisterMode::MISSING)?;
    }
    eprintln!(
        "hybrid-prepare: overlaid {n} + registered {reg} cumulative pages in {}us (off the blackout)",
        t0.elapsed().as_micros()
    );
    Ok(PreparedHybrid {
        guest_memory,
        uffd,
        backend_mappings,
        base: base_clone,
        page_size,
        region_shape,
    })
}

/// Phase B: finish a prepared Hybrid restore at cutover. Overlays/registers ONLY this
/// request's dirty set (the final-round delta), then performs the deferred handler
/// handshake. The caller proceeds to the normal device/vCPU restore with the result.
fn finish_prepared_hybrid(
    prepared: PreparedHybrid,
    mem_backend: &MemBackendConfig,
    mem_state: &GuestMemoryState,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromHybridError> {
    if mem_backend.eager_delta_path.is_some()
        || mem_backend.eager_pages_path.is_some()
        || mem_backend.prefill_pages_path.is_some()
    {
        return Err(GuestMemoryFromHybridError::InvalidPhase);
    }
    // The final snapshot must describe the same memory layout the prepare mapped.
    let shape: Vec<(u64, usize)> = mem_state.regions().map(|r| (r.0 .0, r.1)).collect();
    if shape != prepared.region_shape {
        return Err(GuestMemoryFromHybridError::PreparedMismatch);
    }
    let t0 = std::time::Instant::now();
    let PreparedHybrid {
        guest_memory,
        uffd,
        backend_mappings,
        base,
        page_size,
        ..
    } = prepared;
    let (mut n, mut reg) = (0u64, 0u64);
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        n = overlay_anon_ranges(&backend_mappings, dirty_path, page_size)?;
        reg = register_minor_ranges(&uffd, &backend_mappings, dirty_path, page_size, RegisterMode::MISSING)?;
    }
    // Deferred handshake: mappings + [uffd fd, shared-base fd] to the (now running)
    // handler — identical wire format to the one-shot path.
    let socket = UnixStream::connect(&mem_backend.backend_path)?;
    let json = serde_json::to_string(&backend_mappings).unwrap();
    socket
        .send_with_fds(&[json.as_bytes()], &[uffd.as_raw_fd(), base.as_raw_fd()])
        .map_err(GuestMemoryFromHybridError::Send)?;
    forget(socket);
    eprintln!(
        "hybrid-resume(prepared): delta overlay {n} + register {reg} pages + handshake in {}us",
        t0.elapsed().as_micros()
    );
    Ok((guest_memory, Some(uffd)))
}

/// Hybrid restore: back guest RAM with a memfd seeded from the base image
/// (kernel-served, no fault tax) and register UFFD_MINOR over only the dirty
/// pages, served by an external handler via UFFDIO_CONTINUE.
fn guest_memory_from_hybrid(
    mem_backend: &MemBackendConfig,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
    huge_pages: HugePageConfig,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromHybridError> {
    let base_path = mem_backend
        .base_mem_path
        .as_ref()
        .ok_or(GuestMemoryFromHybridError::MissingBase)?;

    // 1. Map ONE shared shmem base (a tmpfs file, e.g. /dev/shm/base.mem) MAP_PRIVATE.
    //    Every VM on the node maps the SAME base file, so the clean base pages are
    //    shared physical pages (COW on write) — one base copy node-wide instead of a
    //    per-VM 4GB memfd. base_mem_path MUST be tmpfs/shmem so UFFD_MINOR can later
    //    register over the dirty tail. The file is seeded once per node externally.
    let t0 = std::time::Instant::now(); // phase timing: printed at the end (stderr)
    let regions: Vec<_> = mem_state.regions().collect();
    let base = File::open(base_path)?;
    let base_fd = base.as_raw_fd();
    let guest_memory = memory::snapshot_file(base, regions.iter().copied(), track_dirty_pages)?;
    let t_map = t0.elapsed().as_micros();

    let mut backend_mappings = Vec::with_capacity(guest_memory.len());
    let mut offset = 0u64;
    for region in guest_memory.iter() {
        #[allow(deprecated)]
        backend_mappings.push(GuestRegionUffdMapping {
            base_host_virt_addr: region.as_ptr() as u64,
            size: region.size(),
            offset,
            page_size: huge_pages.page_size(),
            page_size_kib: huge_pages.page_size(),
        });
        offset += region.size() as u64;
    }

    let page_size = huge_pages.page_size();

    // Residual = the pages served lazily over UFFD. Load it first so the eager-apply /
    // prefill steps stay DISJOINT from it: an eager-applied or prefaulted residual page
    // would have its PTE present and never fault, so the handler could never supply its
    // fresh content (the guest would read stale base).
    let residual: HashSet<u64> = match mem_backend.dirty_pages_path.as_ref() {
        Some(p) => read_page_offsets(p, page_size as u64)?.into_iter().collect(),
        None => HashSet::new(),
    };

    // 2. ISOLATE the residual into PER-VM ANONYMOUS PRIVATE pages, then serve them with
    //    UFFD MISSING + UFFDIO_COPY. The base shmem is mapped MAP_PRIVATE and shared
    //    node-wide; the old scheme served residual with MINOR/CONTINUE, which writes the
    //    SHARED base pagecache — polluting every other VM mapping that base (and the next
    //    migration's prefill, which reads the base). Overlaying the residual ranges with
    //    MAP_FIXED|MAP_ANONYMOUS|MAP_PRIVATE makes them per-VM from the first byte, so
    //    the handler's COPY lands in this VM's own page and the shared base stays
    //    PRISTINE — required to run hundreds of isolated VMs off one base. Clean pages
    //    are untouched (still kernel-served, shared, COW). Must precede prefill/eager so
    //    those steps see the final backing.
    let t_pre_overlay = t0.elapsed().as_micros();
    let mut n_overlaid = 0u64;
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        let n = overlay_anon_ranges(&backend_mappings, dirty_path, page_size)?;
        info!("hybrid: isolated {n} residual pages into per-VM anonymous memory");
        n_overlaid = n;
    }
    let t_overlay = t0.elapsed().as_micros();

    // 3. Working-set prefill: make the hot BASE pages resident before resume so the
    //    guest's first touches don't lazy-fault. Residual pages are skipped (now anon).
    if let Some(manifest) = mem_backend.prefill_pages_path.as_ref() {
        let n = prefault_pages(&guest_memory, manifest, &residual, page_size)?;
        info!("hybrid: prefaulted {n} working-set pages");
    }

    // 4. Eager-apply the trusted delta (rounds-shipped, already local) into private
    //    memory before resume — these need no UFFD. Disjoint from residual by caller.
    if let (Some(delta), Some(pages)) = (
        mem_backend.eager_delta_path.as_ref(),
        mem_backend.eager_pages_path.as_ref(),
    ) {
        let n = apply_delta(&guest_memory, delta, pages)?;
        info!("hybrid: eager-applied {n} trusted delta pages");
    }

    // 5. UFFD MISSING over ONLY the residual (now anonymous); everything else is base or
    //    eager-applied. The handler resolves these with UFFDIO_COPY into the per-VM page.
    let t_pre_reg = t0.elapsed().as_micros();
    let uffd = create_missing_uffd()?;
    let mut n_registered = 0u64;
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        n_registered = register_minor_ranges(
            &uffd, &backend_mappings, dirty_path, page_size, RegisterMode::MISSING,
        )?;
    }
    let t_reg = t0.elapsed().as_micros();

    // 6. handshake: mappings + [uffd fd, shared-base fd] to the handler. The base fd
    //    is sent for protocol compat; the COPY-based handler does not write into it.
    let socket = UnixStream::connect(&mem_backend.backend_path)?;
    let json = serde_json::to_string(&backend_mappings).unwrap();
    socket
        .send_with_fds(&[json.as_bytes()], &[uffd.as_raw_fd(), base_fd])
        .map_err(GuestMemoryFromHybridError::Send)?;
    forget(socket);

    // Phase breakdown to stderr (collected in the agent's children log): sizes the
    // candidate win of moving overlay+register out of the resume blackout.
    eprintln!(
        "hybrid-load timing: map={}us overlay={}us({} pages) eager+prefill={}us register={}us({} pages) handshake={}us total={}us",
        t_map,
        t_overlay - t_pre_overlay,
        n_overlaid,
        t_pre_reg - t_overlay,
        t_reg - t_pre_reg,
        n_registered,
        t0.elapsed().as_micros() - t_reg,
        t0.elapsed().as_micros()
    );

    Ok((guest_memory, Some(uffd)))
}

/// Precopy restore: map the shared base (`base_mem_path`, works on ext4 — no shmem
/// needed) MAP_PRIVATE and eagerly write the dirty delta into guest memory before
/// resume — NO UFFD. Clean base pages stay shared COW across VMs (lazily cached);
/// applied delta pages become private. The trade vs Hybrid: the (small) delta must
/// be present before resume instead of faulted on demand.
fn guest_memory_from_precopy(
    mem_backend: &MemBackendConfig,
    mem_state: &GuestMemoryState,
    track_dirty_pages: bool,
) -> Result<(Vec<GuestRegionMmap>, Option<Uffd>), GuestMemoryFromHybridError> {
    let base_path = mem_backend
        .base_mem_path
        .as_ref()
        .ok_or(GuestMemoryFromHybridError::MissingBase)?;
    let regions: Vec<_> = mem_state.regions().collect();
    let base = File::open(base_path)?;
    let guest_memory = memory::snapshot_file(base, regions.iter().copied(), track_dirty_pages)?;
    if let Some(dirty_path) = mem_backend.dirty_pages_path.as_ref() {
        apply_delta(&guest_memory, &mem_backend.backend_path, dirty_path)?;
    }
    Ok((guest_memory, None))
}

/// Write each dirty page (from `delta_path` at its file offset) into the MAP_PRIVATE
/// guest memory — COW's just that page private, leaving the shared base pristine.
/// Pure host-side memcpy, runs before resume.
fn apply_delta(
    guest_memory: &[GuestRegionMmap],
    delta_path: &Path,
    dirty_path: &Path,
) -> Result<u64, GuestMemoryFromHybridError> {
    const PS: usize = 4096;
    /// Cap per coalesced run (and per-thread scratch buffer). Large enough that the
    /// pread syscall cost amortizes away, small enough to keep scratch cheap.
    const MAX_RUN: usize = 4 << 20;
    let delta = File::open(delta_path)?;
    let dfd = delta.as_raw_fd();
    // cumulative (file_offset, host_addr, size) per region
    let mut cum = Vec::with_capacity(guest_memory.len());
    let mut off = 0u64;
    for r in guest_memory {
        cum.push((off, r.as_ptr() as u64, r.size() as u64));
        off += r.size() as u64;
    }
    let mut offsets: Vec<u64> = io::BufReader::new(File::open(dirty_path)?)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .map(|o| o & !((PS as u64) - 1))
        .collect();
    offsets.sort_unstable();
    offsets.dedup();

    // The old shape did one 4 KiB pread + memcpy PER PAGE, single-threaded — ~165k
    // syscalls and ~850ms for a busy-Chrome eager set, all inside the resume
    // blackout. Deltas are run-heavy, so: (1) coalesce the sorted offsets into
    // contiguous runs within a region, then (2) apply the runs on a few scoped
    // threads. Destinations are disjoint pages and pread is positioned, so the
    // workers share nothing but the fd.
    struct Run {
        file_off: u64,
        host: u64,
        len: usize,
    }
    let mut runs: Vec<Run> = Vec::new();
    for o in offsets {
        let Some(&(c, host, _)) = cum.iter().find(|&&(c, _, size)| o >= c && o < c + size)
        else {
            continue;
        };
        let host_addr = host + (o - c);
        if let Some(last) = runs.last_mut() {
            // extend only when BOTH file and host are contiguous: adjacent regions
            // are contiguous in file space but not in host space (a run crossing a
            // region boundary would write past the first region's mapping).
            if last.file_off + last.len as u64 == o
                && last.host + last.len as u64 == host_addr
                && last.len + PS <= MAX_RUN
            {
                last.len += PS;
                continue;
            }
        }
        runs.push(Run {
            file_off: o,
            host: host_addr,
            len: PS,
        });
    }

    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(8)
        .min(runs.len().max(1));
    let applied = std::sync::atomic::AtomicU64::new(0);
    let apply_chunk = |chunk: &[Run]| {
        let mut buf = vec![0u8; MAX_RUN];
        for run in chunk {
            // read the fresh pages from the delta file at their guest-memory offsets
            let n = unsafe {
                libc::pread(dfd, buf.as_mut_ptr().cast(), run.len, run.file_off as i64)
            };
            if n <= 0 {
                continue; // beyond EOF / error: nothing to overlay
            }
            // apply only the complete pages read (a short read near EOF mirrors the
            // old per-page `n != PS -> skip` behaviour for the truncated tail)
            let full = (n as usize / PS) * PS;
            if full == 0 {
                continue;
            }
            let dst = run.host as *mut u8;
            // SAFETY: dst..dst+full is within this region's MAP_PRIVATE mapping
            // (runs never cross regions); the writes COW the pages. No two runs
            // overlap (offsets are sorted + deduped), so workers never alias.
            unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, full) };
            applied.fetch_add((full / PS) as u64, std::sync::atomic::Ordering::Relaxed);
        }
    };
    if workers <= 1 {
        apply_chunk(&runs);
    } else {
        let chunk_size = runs.len().div_ceil(workers);
        std::thread::scope(|s| {
            for chunk in runs.chunks(chunk_size) {
                s.spawn(|| apply_chunk(chunk));
            }
        });
    }
    Ok(applied.into_inner())
}

// Retained for the MINOR/CONTINUE back-compat path (e.g. MISSING-unsupported backings);
// the neko Hybrid path now isolates residual into anonymous memory + MISSING, so this is
// not currently wired.
#[allow(dead_code)]
fn create_minor_uffd() -> Result<Uffd, GuestMemoryFromHybridError> {
    // SAFETY: simple syscall wrappers; the fd is owned by the returned Uffd.
    let raw = unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK) };
    if raw < 0 {
        return Err(GuestMemoryFromHybridError::Uffd(io::Error::last_os_error()));
    }
    let raw = raw as RawFd;
    let mut api = UffdioApi {
        api: UFFD_API,
        features: UFFD_FEATURE_MINOR_SHMEM | UFFD_FEATURE_EVENT_REMOVE,
        ioctls: 0,
    };
    // `UFFDIO_API` is u64; ioctl's request arg is c_int on musl and c_ulong on glibc,
    // so cast to the platform's type to build under both libc targets.
    let ret = unsafe { libc::ioctl(raw, UFFDIO_API as _, &mut api as *mut UffdioApi) };
    if ret != 0 {
        let err = io::Error::last_os_error();
        unsafe { libc::close(raw) };
        return Err(GuestMemoryFromHybridError::Uffd(err));
    }
    Ok(unsafe { Uffd::from_raw_fd(raw) })
}

/// UFFD for MISSING-mode faults over the (now anonymous) residual pages — resolved by
/// the handler with UFFDIO_COPY into the VM's own private page. No MINOR_SHMEM feature:
/// MISSING is the default fault mode and needs no shmem-specific capability.
fn create_missing_uffd() -> Result<Uffd, GuestMemoryFromHybridError> {
    // SAFETY: simple syscall wrappers; the fd is owned by the returned Uffd.
    let raw = unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK) };
    if raw < 0 {
        return Err(GuestMemoryFromHybridError::Uffd(io::Error::last_os_error()));
    }
    let raw = raw as RawFd;
    let mut api = UffdioApi {
        api: UFFD_API,
        features: UFFD_FEATURE_EVENT_REMOVE,
        ioctls: 0,
    };
    let ret = unsafe { libc::ioctl(raw, UFFDIO_API as _, &mut api as *mut UffdioApi) };
    if ret != 0 {
        let err = io::Error::last_os_error();
        unsafe { libc::close(raw) };
        return Err(GuestMemoryFromHybridError::Uffd(err));
    }
    Ok(unsafe { Uffd::from_raw_fd(raw) })
}

fn register_minor_ranges(
    uffd: &Uffd,
    mappings: &[GuestRegionUffdMapping],
    dirty_path: &Path,
    page_size: usize,
    mode: RegisterMode,
) -> Result<u64, GuestMemoryFromHybridError> {
    let ps = page_size as u64;
    let file = File::open(dirty_path)?;
    let mut offsets: Vec<u64> = io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .map(|o| o & !(ps - 1))
        .collect();
    offsets.sort_unstable();
    offsets.dedup();

    let mut registered = 0u64;
    let mut i = 0;
    while i < offsets.len() {
        // coalesce a run of consecutive pages
        let start = offsets[i];
        let mut end = start + ps;
        let mut j = i + 1;
        while j < offsets.len() && offsets[j] == end {
            end += ps;
            j += 1;
        }
        if let Some(m) = mappings
            .iter()
            .find(|m| start >= m.offset && start < m.offset + m.size as u64)
        {
            let end = end.min(m.offset + m.size as u64); // clamp to this region
            let addr = (m.base_host_virt_addr + (start - m.offset)) as *mut libc::c_void;
            uffd.register_with_mode(addr, (end - start) as usize, mode)
                .map_err(GuestMemoryFromHybridError::Register)?;
            registered += (end - start) / ps;
        }
        i = j;
    }
    Ok(registered)
}

/// Replace the residual page ranges (from `dirty_path`) with MAP_FIXED anonymous
/// PRIVATE memory, so each VM's residual is its own per-VM page from the first byte.
/// This is what lets the handler resolve them with UFFDIO_COPY (writing the VM's own
/// page) instead of MINOR/CONTINUE (writing the SHARED base pagecache, which would
/// pollute every other VM on the node and the next migration's prefill). Clean pages
/// stay backed by the shared base. Coalesces consecutive pages to minimize mmap calls.
fn overlay_anon_ranges(
    mappings: &[GuestRegionUffdMapping],
    dirty_path: &Path,
    page_size: usize,
) -> Result<u64, GuestMemoryFromHybridError> {
    let ps = page_size as u64;
    let mut offsets: Vec<u64> = io::BufReader::new(File::open(dirty_path)?)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .map(|o| o & !(ps - 1))
        .collect();
    offsets.sort_unstable();
    offsets.dedup();

    let mut overlaid = 0u64;
    let mut i = 0;
    while i < offsets.len() {
        let start = offsets[i];
        let mut end = start + ps;
        let mut j = i + 1;
        while j < offsets.len() && offsets[j] == end {
            end += ps;
            j += 1;
        }
        if let Some(m) = mappings
            .iter()
            .find(|m| start >= m.offset && start < m.offset + m.size as u64)
        {
            let end = end.min(m.offset + m.size as u64); // clamp to this region
            let addr = (m.base_host_virt_addr + (start - m.offset)) as *mut libc::c_void;
            // SAFETY: addr..end lies within this region's existing guest-memory mapping;
            // MAP_FIXED atomically replaces that sub-range with fresh anonymous private
            // pages at the same address. The guest VA layout is unchanged.
            let p = unsafe {
                libc::mmap(
                    addr,
                    (end - start) as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_FIXED | libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(GuestMemoryFromHybridError::Io(io::Error::last_os_error()));
            }
            overlaid += (end - start) / ps;
        }
        i = j;
    }
    Ok(overlaid)
}

/// Read a page-offset file (one u64 per line), page-align, sort + dedup.
fn read_page_offsets(path: &Path, ps: u64) -> Result<Vec<u64>, GuestMemoryFromHybridError> {
    let file = File::open(path)?;
    let mut offsets: Vec<u64> = io::BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .map(|o| o & !(ps - 1))
        .collect();
    offsets.sort_unstable();
    offsets.dedup();
    Ok(offsets)
}

/// Prefault (make resident) the working-set pages in `manifest` by reading one byte
/// of each from the MAP_PRIVATE base — a read brings in the shared page-cache page
/// (no COW), so the guest's first touch doesn't lazy-fault. Pages in `skip` (the
/// residual/UFFD set) are left untouched so they still minor-fault.
fn prefault_pages(
    guest_memory: &[GuestRegionMmap],
    manifest: &Path,
    skip: &HashSet<u64>,
    page_size: usize,
) -> Result<u64, GuestMemoryFromHybridError> {
    let ps = page_size as u64;
    // cumulative (file_offset, host_addr, size) per region
    let mut cum = Vec::with_capacity(guest_memory.len());
    let mut off = 0u64;
    for r in guest_memory {
        cum.push((off, r.as_ptr() as u64, r.size() as u64));
        off += r.size() as u64;
    }
    let mut n = 0u64;
    for o in read_page_offsets(manifest, ps)? {
        if skip.contains(&o) {
            continue;
        }
        if let Some(&(c, host, _)) = cum.iter().find(|&&(c, _, size)| o >= c && o < c + size) {
            let addr = (host + (o - c)) as *const u8;
            // SAFETY: addr is within this region's MAP_PRIVATE mapping; a volatile read
            // faults the page in read-only (shared base page cache, no COW).
            unsafe { std::ptr::read_volatile(addr) };
            n += 1;
        }
    }
    Ok(n)
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;

    use vmm_sys_util::tempfile::TempFile;

    use super::*;
    use crate::Vmm;
    #[cfg(target_arch = "x86_64")]
    use crate::builder::tests::insert_vmclock_device;
    #[cfg(target_arch = "x86_64")]
    use crate::builder::tests::insert_vmgenid_device;
    use crate::builder::tests::{
        CustomBlockConfig, default_kernel_cmdline, default_vmm, insert_balloon_device,
        insert_block_devices, insert_net_device, insert_vsock_device,
    };
    #[cfg(target_arch = "aarch64")]
    use crate::construct_kvm_mpidrs;
    use crate::devices::virtio::block::CacheType;
    use crate::snapshot::Persist;
    use crate::vmm_config::balloon::BalloonDeviceConfig;
    use crate::vmm_config::net::NetworkInterfaceConfig;
    use crate::vmm_config::vsock::tests::default_config;
    use crate::vstate::memory::{GuestMemoryRegionState, GuestRegionType};

    #[test]
    fn test_read_page_offsets_aligns_sorts_dedups() {
        // The page-offset parser backs both the residual (UFFD) set and the
        // prefill/eager sets, so alignment + sort + dedup must be exact.
        let tf = TempFile::new().unwrap();
        // unsorted, with a non-aligned offset (4097 -> 4096) and a duplicate (8192)
        std::fs::write(tf.as_path(), "8192\n4097\n4096\n8192\n0\nnot_a_number\n").unwrap();
        let got = read_page_offsets(tf.as_path(), 4096).unwrap();
        assert_eq!(got, vec![0, 4096, 8192]);
    }

    #[test]
    fn test_mem_backend_config_parses_fusion_fields() {
        // The fused Hybrid body adds eager_delta/eager_pages/prefill paths; absent
        // ones must default to None (so the plain Hybrid/File paths are unaffected).
        let json = r#"{"backend_path":"/tmp/u.sock","backend_type":"Hybrid",
            "base_mem_path":"/dev/shm/neko.mem","dirty_pages_path":"/x/residual.set",
            "eager_delta_path":"/x/buf","eager_pages_path":"/x/trusted.set",
            "prefill_pages_path":"/x/warmset"}"#;
        let cfg: crate::vmm_config::snapshot::MemBackendConfig =
            serde_json::from_str(json).unwrap();
        assert_eq!(cfg.eager_delta_path.unwrap().to_str().unwrap(), "/x/buf");
        assert_eq!(cfg.eager_pages_path.unwrap().to_str().unwrap(), "/x/trusted.set");
        assert_eq!(cfg.prefill_pages_path.unwrap().to_str().unwrap(), "/x/warmset");

        let minimal = r#"{"backend_path":"/m","backend_type":"File"}"#;
        let cfg2: crate::vmm_config::snapshot::MemBackendConfig =
            serde_json::from_str(minimal).unwrap();
        assert!(cfg2.eager_delta_path.is_none());
        assert!(cfg2.prefill_pages_path.is_none());
    }

    fn default_vmm_with_devices() -> Vmm {
        let mut event_manager = EventManager::new().expect("Cannot create EventManager");
        let mut vmm = default_vmm();
        let mut cmdline = default_kernel_cmdline();

        // Add a balloon device.
        let balloon_config = BalloonDeviceConfig {
            amount_mib: 0,
            deflate_on_oom: false,
            stats_polling_interval_s: 0,
            free_page_hinting: false,
            free_page_reporting: false,
        };
        insert_balloon_device(&mut vmm, &mut cmdline, &mut event_manager, balloon_config);

        // Add a block device.
        let drive_id = String::from("root");
        let block_configs = vec![CustomBlockConfig::new(
            drive_id,
            true,
            None,
            true,
            CacheType::Unsafe,
        )];
        insert_block_devices(&mut vmm, &mut cmdline, &mut event_manager, block_configs);

        // Add net device.
        let network_interface = NetworkInterfaceConfig {
            iface_id: String::from("netif"),
            host_dev_name: String::from("hostname"),
            guest_mac: None,
            mtu: None,
            rx_rate_limiter: None,
            tx_rate_limiter: None,
        };
        insert_net_device(
            &mut vmm,
            &mut cmdline,
            &mut event_manager,
            network_interface,
        );

        // Add vsock device.
        let mut tmp_sock_file = TempFile::new().unwrap();
        tmp_sock_file.remove().unwrap();
        let vsock_config = default_config(&tmp_sock_file);

        insert_vsock_device(&mut vmm, &mut cmdline, &mut event_manager, vsock_config);

        #[cfg(target_arch = "x86_64")]
        insert_vmgenid_device(&mut vmm);
        #[cfg(target_arch = "x86_64")]
        insert_vmclock_device(&mut vmm);

        vmm
    }

    #[test]
    fn test_microvm_state_snapshot() {
        let vmm = default_vmm_with_devices();
        let states = vmm.device_manager.save();

        // Only checking that all devices are saved, actual device state
        // is tested by that device's tests.
        assert_eq!(states.mmio_state.block_devices.len(), 1);
        assert_eq!(states.mmio_state.net_devices.len(), 1);
        assert!(states.mmio_state.vsock_device.is_some());
        assert!(states.mmio_state.balloon_device.is_some());

        let vcpu_states = vec![VcpuState::default()];
        #[cfg(target_arch = "aarch64")]
        let mpidrs = construct_kvm_mpidrs(&vcpu_states);
        let microvm_state = MicrovmState {
            device_states: states,
            vcpu_states,
            kvm_state: Default::default(),
            vm_info: VmInfo {
                mem_size_mib: 1u64,
                ..Default::default()
            },
            #[cfg(target_arch = "aarch64")]
            vm_state: vmm.vm.as_kvm().unwrap().save_state(&mpidrs).unwrap(),
            #[cfg(target_arch = "x86_64")]
            vm_state: vmm.vm.as_kvm().unwrap().save_state().unwrap(),
        };

        let serialized_data = bitcode::serialize(&microvm_state).unwrap();

        let restored_microvm_state: MicrovmState = bitcode::deserialize(&serialized_data).unwrap();

        assert_eq!(restored_microvm_state.vm_info, microvm_state.vm_info);
        assert_eq!(
            restored_microvm_state.device_states.mmio_state,
            microvm_state.device_states.mmio_state
        )
    }

    #[test]
    fn test_create_guest_memory() {
        let mem_state = GuestMemoryState {
            regions: vec![GuestMemoryRegionState {
                base_address: 0,
                size: 0x20000,
                region_type: GuestRegionType::Dram,
                plugged: vec![true],
            }],
        };

        let (_, uffd_regions) =
            create_guest_memory(&mem_state, false, HugePageConfig::None).unwrap();

        assert_eq!(uffd_regions.len(), 1);
        assert_eq!(uffd_regions[0].size, 0x20000);
        assert_eq!(uffd_regions[0].offset, 0);
        assert_eq!(uffd_regions[0].page_size, HugePageConfig::None.page_size());
    }

    #[test]
    fn test_send_uffd_handshake() {
        #[allow(deprecated)]
        let uffd_regions = vec![
            GuestRegionUffdMapping {
                base_host_virt_addr: 0,
                size: 0x100000,
                offset: 0,
                page_size: HugePageConfig::None.page_size(),
                page_size_kib: HugePageConfig::None.page_size(),
            },
            GuestRegionUffdMapping {
                base_host_virt_addr: 0x100000,
                size: 0x200000,
                offset: 0,
                page_size: HugePageConfig::Hugetlbfs2M.page_size(),
                page_size_kib: HugePageConfig::Hugetlbfs2M.page_size(),
            },
        ];

        let uds_path = TempFile::new().unwrap();
        let uds_path = uds_path.as_path();
        std::fs::remove_file(uds_path).unwrap();

        let listener = UnixListener::bind(uds_path).expect("Cannot bind to socket path");

        send_uffd_handshake(uds_path, &uffd_regions, &std::io::stdin()).unwrap();

        let (stream, _) = listener.accept().expect("Cannot listen on UDS socket");

        let mut message_buf = vec![0u8; 1024];
        let (bytes_read, _) = stream
            .recv_with_fd(&mut message_buf[..])
            .expect("Cannot recv_with_fd");
        message_buf.resize(bytes_read, 0);

        let deserialized: Vec<GuestRegionUffdMapping> =
            serde_json::from_slice(&message_buf).unwrap();

        assert_eq!(uffd_regions, deserialized);
    }
}
