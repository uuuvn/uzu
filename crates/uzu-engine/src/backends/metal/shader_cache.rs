//! Runtime control over Metal's on-disk shader cache.
//!
//! Pipeline state creation is cached by Metal on disk (per user, shared across
//! processes) in addition to the explicit in-process caches in
//! [`MetalContext`](super::MetalContext). With a warm on-disk cache, creating a
//! pipeline state from a function is tens of times faster than on the very
//! first run, which makes cold-start behavior impossible to reproduce.
//!
//! Setting the `UZU_CLEAN_COMPILE` environment variable redirects Metal's
//! on-disk cache to a fresh empty directory for the duration of the process,
//! reproducing a first-ever run without touching the user's real cache.

use std::{
    ffi::{CString, c_char, c_int, c_void},
    mem::transmute,
    path::PathBuf,
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

use obfstr::obfstr;
use objc2::{ffi::objc_msgSend, runtime::AnyClass, sel};

/// Name of the environment variable that enables the on-disk shader cache bypass.
pub(super) const CLEAN_COMPILE_ENV_VAR: &str = "UZU_CLEAN_COMPILE";

const RTLD_NOW: c_int = 2;

unsafe extern "C" {
    fn dlopen(
        path: *const c_char,
        mode: c_int,
    ) -> *mut c_void;
    fn dlsym(
        handle: *mut c_void,
        symbol: *const c_char,
    ) -> *mut c_void;
}

/// Redirects Metal's on-disk shader cache to a fresh empty directory when
/// [`CLEAN_COMPILE_ENV_VAR`] is set, so that pipeline state creation behaves as
/// on a first-ever run. Runs at most once per process and must happen before
/// the first Metal device or library is created: Metal initializes its cache
/// lazily and ignores later redirection attempts.
pub(super) fn bypass_if_requested() {
    static BYPASSED: OnceLock<()> = OnceLock::new();
    BYPASSED.get_or_init(|| {
        if std::env::var(CLEAN_COMPILE_ENV_VAR).is_err() {
            return;
        }
        match redirect_on_disk_cache() {
            Ok(directory) => {
                eprintln!(
                    "uzu: {CLEAN_COMPILE_ENV_VAR} is set, Metal's on-disk shader cache is redirected to the fresh \
                     directory {} for this process; pipeline state creation will behave as on a first-ever run",
                    directory.display(),
                );
            },
            Err(error) => {
                eprintln!(
                    "uzu: {CLEAN_COMPILE_ENV_VAR} is set, but Metal's on-disk shader cache could not be redirected: \
                     {error}. The cache stays active, pipeline state creation may be faster than on a first-ever run",
                );
            },
        }
    });
}

fn redirect_on_disk_cache() -> Result<PathBuf, String> {
    // `MTLSetShaderCachePath` is a private Metal entry point that overrides the
    // location of the on-disk shader cache (`com.apple.metal`).
    let entry_point = metal_entry_point(obfstr!("MTLSetShaderCachePath"))
        .ok_or_else(|| "MTLSetShaderCachePath is not available".to_string())?;

    let directory = fresh_cache_directory()?;
    let path = directory.to_str().ok_or_else(|| format!("non-UTF-8 path {}", directory.display()))?;
    let path = retained_ns_string(path).ok_or_else(|| "cannot create NSString".to_string())?;

    let set_shader_cache_path: unsafe extern "C" fn(*mut c_void) = unsafe { transmute(entry_point) };
    unsafe { set_shader_cache_path(path) };

    Ok(directory)
}

/// Looks up a private Metal.framework entry point at runtime, so the symbol is
/// never referenced directly (same reasoning as the private selectors in
/// [`super::metal_extensions::DeviceExt`]) and its absence is a runtime error
/// instead of a link failure.
fn metal_entry_point(name: &str) -> Option<*mut c_void> {
    // Metal.framework is already linked into the process, so `dlopen` just
    // returns the existing handle. It is intentionally never `dlclose`d.
    let framework_path =
        CString::new("/System/Library/Frameworks/Metal.framework/Metal").expect("framework path has no NUL byte");
    let handle = unsafe { dlopen(framework_path.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        return None;
    }
    let name = CString::new(name).expect("entry point name has no interior NUL byte");
    let entry_point = unsafe { dlsym(handle, name.as_ptr()) };
    (!entry_point.is_null()).then_some(entry_point)
}

/// Creates a retained `NSString`. The object is intentionally never released:
/// it is needed once per process, and autoreleasing would require an
/// autorelease pool that does not exist outside of Objective-C run loops.
fn retained_ns_string(value: &str) -> Option<*mut c_void> {
    let class = AnyClass::get(c"NSString")?;
    let value = CString::new(value).ok()?;

    let alloc: unsafe extern "C" fn(*const AnyClass, objc2::runtime::Sel) -> *mut c_void =
        unsafe { transmute(objc_msgSend as *const ()) };
    let init: unsafe extern "C" fn(*mut c_void, objc2::runtime::Sel, *const c_char) -> *mut c_void =
        unsafe { transmute(objc_msgSend as *const ()) };

    let object = unsafe { alloc(class, sel!(alloc)) };
    if object.is_null() {
        return None;
    }
    let object = unsafe { init(object, sel!(initWithUTF8String:), value.as_ptr()) };
    (!object.is_null()).then_some(object)
}

fn fresh_cache_directory() -> Result<PathBuf, String> {
    let unique = SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_nanos()).unwrap_or(0);
    let directory = std::env::temp_dir().join(format!("uzu-clean-compile-{}-{unique}", std::process::id()));
    // The directory is left behind on process exit: Metal's compiler service
    // may outlive the process and keep writing into it. It lives in the
    // per-user temporary directory, which the operating system cleans up.
    std::fs::create_dir_all(&directory).map_err(|error| format!("cannot create {}: {error}", directory.display()))?;
    Ok(directory)
}
