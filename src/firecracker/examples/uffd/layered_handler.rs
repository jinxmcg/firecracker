use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

use serde::Deserialize;
use userfaultfd::{Error, Event, Uffd};
use vmm_sys_util::sock_ctrl_msg::ScmSocket;

const SEEK_DATA: i32 = 3;
const PAGE_SIZE: usize = 4096;

#[derive(Clone, Debug, Deserialize)]
struct GuestRegionUffdMapping {
    base_host_virt_addr: u64,
    size: usize,
    offset: u64,
    page_size: usize,
}

impl GuestRegionUffdMapping {
    fn contains(&self, fault_page_addr: u64) -> bool {
        fault_page_addr >= self.base_host_virt_addr
            && fault_page_addr < self.base_host_virt_addr + self.size as u64
    }
}

struct LayeredHandler {
    mappings: Vec<GuestRegionUffdMapping>,
    uffd: Uffd,
    base: File,
    delta: File,
    page_size: usize,
    page: Vec<u8>,
    faults: u64,
    delta_hits: u64,
    base_hits: u64,
    prefill: bool,
    prefill_region: usize,
    prefill_offset: usize,
}

impl LayeredHandler {
    fn from_stream(stream: &UnixStream, base: File, delta: File) -> Self {
        let (body, uffd_file) = get_mappings_and_file(stream);
        let mappings = serde_json::from_str::<Vec<GuestRegionUffdMapping>>(&body)
            .unwrap_or_else(|_| panic!("cannot deserialize UFFD mappings: {body}"));
        let page_size = mappings
            .first()
            .expect("Firecracker sent no UFFD mappings")
            .page_size;
        assert!(page_size.is_power_of_two());

        let uffd = unsafe { Uffd::from_raw_fd(uffd_file.into_raw_fd()) };

        Self {
            mappings,
            uffd,
            base,
            delta,
            page_size,
            page: vec![0; page_size.max(PAGE_SIZE)],
            faults: 0,
            delta_hits: 0,
            base_hits: 0,
            prefill: std::env::var_os("UFFD_PREFILL_DISABLE").is_none(),
            prefill_region: 0,
            prefill_offset: 0,
        }
    }

    fn run(&mut self) {
        loop {
            let Some(event) = self.uffd.read_event().expect("failed to read UFFD event") else {
                if self.prefill && self.prefill_some(64) {
                    continue;
                }
                if self.prefill && self.prefill_region >= self.mappings.len() {
                    self.unregister_all();
                    eprintln!(
                        "uffd prefill complete faults={} delta_hits={} base_hits={}",
                        self.faults, self.delta_hits, self.base_hits
                    );
                    return;
                }
                std::thread::sleep(Duration::from_millis(1));
                continue;
            };

            match event {
                Event::Pagefault { addr, .. } => {
                    if !self.serve_page(addr.cast()) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                Event::Remove { start, end } => self.unregister_range(start, end),
                other => panic!("unexpected UFFD event: {other:?}"),
            }
        }
    }

    fn prefill_some(&mut self, budget_pages: usize) -> bool {
        let mut copied = 0;
        while self.prefill_region < self.mappings.len() && copied < budget_pages {
            let mapping = self.mappings[self.prefill_region].clone();
            if self.prefill_offset >= mapping.size {
                self.prefill_region += 1;
                self.prefill_offset = 0;
                continue;
            }

            let dst = mapping.base_host_virt_addr + self.prefill_offset as u64;
            let offset = mapping.offset + self.prefill_offset as u64;
            let mut page = std::mem::take(&mut self.page);
            let _ = read_layered_page(&mut self.base, &mut self.delta, offset, &mut page);
            self.page = page;
            self.copy_page(dst as *mut c_void);

            self.prefill_offset += self.page_size;
            copied += 1;
        }

        copied > 0
    }

    fn unregister_all(&mut self) {
        for mapping in self.mappings.clone() {
            let start = mapping.base_host_virt_addr as *mut c_void;
            self.uffd
                .unregister(start, mapping.size)
                .expect("UFFD unregister after prefill");
        }
    }

    fn unregister_range(&mut self, start: *mut c_void, end: *mut c_void) {
        assert!(
            (start as usize).is_multiple_of(self.page_size)
                && (end as usize).is_multiple_of(self.page_size)
                && end > start
        );
        let len = unsafe { end.offset_from_unsigned(start) };
        self.uffd.unregister(start, len).expect("UFFD unregister");
    }

    fn serve_page(&mut self, addr: *mut u8) -> bool {
        let dst = (addr as usize & !(self.page_size - 1)) as *mut libc::c_void;
        let fault_page_addr = dst as u64;
        let mapping = self
            .mappings
            .iter()
            .find(|mapping| mapping.contains(fault_page_addr))
            .unwrap_or_else(|| panic!("fault address {addr:?} is outside guest memory mappings"))
            .clone();

        let offset = mapping.offset + fault_page_addr - mapping.base_host_virt_addr;
        let mut page = std::mem::take(&mut self.page);
        let source = read_layered_page(&mut self.base, &mut self.delta, offset, &mut page);
        self.page = page;

        match source {
            PageSource::Delta => self.delta_hits += 1,
            PageSource::Base => self.base_hits += 1,
        }
        self.faults += 1;

        if !self.copy_page(dst) {
            return false;
        }

        if self.faults.is_multiple_of(1024) {
            eprintln!(
                "uffd faults={} delta_hits={} base_hits={}",
                self.faults, self.delta_hits, self.base_hits
            );
        }

        true
    }

    fn copy_page(&self, dst: *mut c_void) -> bool {
        unsafe {
            match self.uffd.copy(self.page.as_ptr().cast(), dst, self.page_size, true) {
                Ok(value) => assert!(value > 0),
                Err(Error::PartiallyCopied(bytes_copied))
                    if bytes_copied == 0 || bytes_copied == (-libc::EAGAIN) as usize =>
                {
                    return false;
                }
                Err(Error::CopyFailed(errno))
                    if std::io::Error::from(errno).raw_os_error().unwrap() == libc::EEXIST => {}
                Err(Error::CopyFailed(errno))
                    if std::io::Error::from(errno).raw_os_error().unwrap() == libc::ESRCH =>
                {
                    eprintln!("UFFD mapping disappeared during copy; exiting");
                    std::process::exit(0);
                }
                Err(err) => panic!("UFFD copy failed: {err:?}"),
            }
        }

        true
    }
}

#[derive(Clone, Copy)]
enum PageSource {
    Base,
    Delta,
}

fn read_layered_page(base: &mut File, delta: &mut File, offset: u64, page: &mut [u8]) -> PageSource {
    page.fill(0);

    if has_data_at(delta, offset) {
        read_exact_at(delta, offset, page);
        return PageSource::Delta;
    }

    read_exact_at(base, offset, page);
    PageSource::Base
}

fn has_data_at(file: &File, offset: u64) -> bool {
    let data = unsafe { libc::lseek(file.as_raw_fd(), offset as libc::off_t, SEEK_DATA) };
    data >= 0 && data as u64 <= offset
}

fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) {
    file.seek(SeekFrom::Start(offset)).expect("seek memory layer");
    file.read_exact(buf).expect("read memory layer");
}

fn get_mappings_and_file(stream: &UnixStream) -> (String, File) {
    for _ in 1..=50 {
        let mut message_buf = vec![0u8; 4096];
        match stream.recv_with_fd(&mut message_buf[..]) {
            Ok((bytes_read, Some(file))) => {
                message_buf.resize(bytes_read, 0);
                return (
                    String::from_utf8(message_buf).expect("Firecracker UFFD body is not UTF-8"),
                    file,
                );
            }
            Ok((bytes_read, None)) => {
                message_buf.resize(bytes_read, 0);
                eprintln!(
                    "received UFFD mapping without fd: {}",
                    String::from_utf8_lossy(&message_buf)
                );
            }
            Err(err) => eprintln!("waiting for Firecracker UFFD handshake: {err}"),
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    panic!("did not receive UFFD fd and mappings from Firecracker");
}

fn main() {
    let mut args = std::env::args_os().skip(1);
    let socket_path = PathBuf::from(args.next().expect("usage: uffd-layered-handler SOCKET BASE_MEM DELTA_MEM"));
    let base_path = PathBuf::from(args.next().expect("missing BASE_MEM"));
    let delta_path = PathBuf::from(args.next().expect("missing DELTA_MEM"));

    let _ = std::fs::remove_file(&socket_path);
    let listener = UnixListener::bind(&socket_path).expect("bind UFFD socket");
    let base = File::open(&base_path).expect("open base memory file");
    let delta = File::open(&delta_path).expect("open delta memory file");

    eprintln!(
        "uffd-layered-handler listening socket={} base={} delta={}",
        socket_path.display(),
        base_path.display(),
        delta_path.display()
    );

    let (stream, _) = listener.accept().expect("accept Firecracker UFFD connection");
    let mut handler = LayeredHandler::from_stream(&stream, base, delta);
    handler.run();
}
