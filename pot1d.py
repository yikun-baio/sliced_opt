
# This Python module `pot1d` is auto-generated using cxx2py tool!
__all__ = []
import ctypes
import rbc

def _load_library(name):
    # FIXME: win
    return ctypes.cdll.LoadLibrary(f'lib{name}.so')

_lib = _load_library("cxx2py_pot1d")

_target_info = rbc.targetinfo.TargetInfo('cpu')


_lib.get_wcscpy_address.argtypes = ()
_lib.get_wcscpy_address.restype = ctypes.c_void_p
with _target_info:
    _wcscpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true)")
wcscpy = _wcscpy_signature.toctypes()(_lib.get_wcscpy_address())
__all__.append("wcscpy")


_lib.get_wcsncpy_address.argtypes = ()
_lib.get_wcsncpy_address.restype = ctypes.c_void_p
with _target_info:
    _wcsncpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wcsncpy = _wcsncpy_signature.toctypes()(_lib.get_wcsncpy_address())
__all__.append("wcsncpy")


_lib.get_wcscat_address.argtypes = ()
_lib.get_wcscat_address.restype = ctypes.c_void_p
with _target_info:
    _wcscat_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true)")
wcscat = _wcscat_signature.toctypes()(_lib.get_wcscat_address())
__all__.append("wcscat")


_lib.get_wcsncat_address.argtypes = ()
_lib.get_wcsncat_address.restype = ctypes.c_void_p
with _target_info:
    _wcsncat_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wcsncat = _wcsncat_signature.toctypes()(_lib.get_wcsncat_address())
__all__.append("wcsncat")


_lib.get_wcscmp_address.argtypes = ()
_lib.get_wcscmp_address.restype = ctypes.c_void_p
with _target_info:
    _wcscmp_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *) noexcept(true)")
wcscmp = _wcscmp_signature.toctypes()(_lib.get_wcscmp_address())
__all__.append("wcscmp")


_lib.get_wcsncmp_address.argtypes = ()
_lib.get_wcsncmp_address.restype = ctypes.c_void_p
with _target_info:
    _wcsncmp_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, size_t) noexcept(true)")
wcsncmp = _wcsncmp_signature.toctypes()(_lib.get_wcsncmp_address())
__all__.append("wcsncmp")


_lib.get_wcscasecmp_address.argtypes = ()
_lib.get_wcscasecmp_address.restype = ctypes.c_void_p
with _target_info:
    _wcscasecmp_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *) noexcept(true)")
wcscasecmp = _wcscasecmp_signature.toctypes()(_lib.get_wcscasecmp_address())
__all__.append("wcscasecmp")


_lib.get_wcsncasecmp_address.argtypes = ()
_lib.get_wcsncasecmp_address.restype = ctypes.c_void_p
with _target_info:
    _wcsncasecmp_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, size_t) noexcept(true)")
wcsncasecmp = _wcsncasecmp_signature.toctypes()(_lib.get_wcsncasecmp_address())
__all__.append("wcsncasecmp")


_lib.get_wcscasecmp_l_address.argtypes = ()
_lib.get_wcscasecmp_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcscasecmp_l_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, locale_t) noexcept(true)")
wcscasecmp_l = _wcscasecmp_l_signature.toctypes()(_lib.get_wcscasecmp_l_address())
__all__.append("wcscasecmp_l")


_lib.get_wcsncasecmp_l_address.argtypes = ()
_lib.get_wcsncasecmp_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcsncasecmp_l_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, size_t, locale_t) noexcept(true)")
wcsncasecmp_l = _wcsncasecmp_l_signature.toctypes()(_lib.get_wcsncasecmp_l_address())
__all__.append("wcsncasecmp_l")


_lib.get_wcscoll_address.argtypes = ()
_lib.get_wcscoll_address.restype = ctypes.c_void_p
with _target_info:
    _wcscoll_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *) noexcept(true)")
wcscoll = _wcscoll_signature.toctypes()(_lib.get_wcscoll_address())
__all__.append("wcscoll")


_lib.get_wcsxfrm_address.argtypes = ()
_lib.get_wcsxfrm_address.restype = ctypes.c_void_p
with _target_info:
    _wcsxfrm_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wcsxfrm = _wcsxfrm_signature.toctypes()(_lib.get_wcsxfrm_address())
__all__.append("wcsxfrm")


_lib.get_wcscoll_l_address.argtypes = ()
_lib.get_wcscoll_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcscoll_l_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, locale_t) noexcept(true)")
wcscoll_l = _wcscoll_l_signature.toctypes()(_lib.get_wcscoll_l_address())
__all__.append("wcscoll_l")


_lib.get_wcsxfrm_l_address.argtypes = ()
_lib.get_wcsxfrm_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcsxfrm_l_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *, const wchar_t *, size_t, locale_t) noexcept(true)")
wcsxfrm_l = _wcsxfrm_l_signature.toctypes()(_lib.get_wcsxfrm_l_address())
__all__.append("wcsxfrm_l")


_lib.get_wcsdup_address.argtypes = ()
_lib.get_wcsdup_address.restype = ctypes.c_void_p
with _target_info:
    _wcsdup_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *) noexcept(true)")
wcsdup = _wcsdup_signature.toctypes()(_lib.get_wcsdup_address())
__all__.append("wcsdup")


_lib.get_wcschr_address.argtypes = ()
_lib.get_wcschr_address.restype = ctypes.c_void_p
with _target_info:
    _wcschr_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, wchar_t) noexcept(true)")
wcschr = _wcschr_signature.toctypes()(_lib.get_wcschr_address())
__all__.append("wcschr")


_lib.get_wcsrchr_address.argtypes = ()
_lib.get_wcsrchr_address.restype = ctypes.c_void_p
with _target_info:
    _wcsrchr_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, wchar_t) noexcept(true)")
wcsrchr = _wcsrchr_signature.toctypes()(_lib.get_wcsrchr_address())
__all__.append("wcsrchr")


_lib.get_wcschrnul_address.argtypes = ()
_lib.get_wcschrnul_address.restype = ctypes.c_void_p
with _target_info:
    _wcschrnul_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, wchar_t) noexcept(true)")
wcschrnul = _wcschrnul_signature.toctypes()(_lib.get_wcschrnul_address())
__all__.append("wcschrnul")


_lib.get_wcscspn_address.argtypes = ()
_lib.get_wcscspn_address.restype = ctypes.c_void_p
with _target_info:
    _wcscspn_signature = rbc.typesystem.Type.fromstring("size_t (const wchar_t *, const wchar_t *) noexcept(true)")
wcscspn = _wcscspn_signature.toctypes()(_lib.get_wcscspn_address())
__all__.append("wcscspn")


_lib.get_wcsspn_address.argtypes = ()
_lib.get_wcsspn_address.restype = ctypes.c_void_p
with _target_info:
    _wcsspn_signature = rbc.typesystem.Type.fromstring("size_t (const wchar_t *, const wchar_t *) noexcept(true)")
wcsspn = _wcsspn_signature.toctypes()(_lib.get_wcsspn_address())
__all__.append("wcsspn")


_lib.get_wcspbrk_address.argtypes = ()
_lib.get_wcspbrk_address.restype = ctypes.c_void_p
with _target_info:
    _wcspbrk_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, const wchar_t *) noexcept(true)")
wcspbrk = _wcspbrk_signature.toctypes()(_lib.get_wcspbrk_address())
__all__.append("wcspbrk")


_lib.get_wcsstr_address.argtypes = ()
_lib.get_wcsstr_address.restype = ctypes.c_void_p
with _target_info:
    _wcsstr_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, const wchar_t *) noexcept(true)")
wcsstr = _wcsstr_signature.toctypes()(_lib.get_wcsstr_address())
__all__.append("wcsstr")


_lib.get_wcstok_address.argtypes = ()
_lib.get_wcstok_address.restype = ctypes.c_void_p
with _target_info:
    _wcstok_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstok = _wcstok_signature.toctypes()(_lib.get_wcstok_address())
__all__.append("wcstok")


_lib.get_wcslen_address.argtypes = ()
_lib.get_wcslen_address.restype = ctypes.c_void_p
with _target_info:
    _wcslen_signature = rbc.typesystem.Type.fromstring("size_t (const wchar_t *) noexcept(true)")
wcslen = _wcslen_signature.toctypes()(_lib.get_wcslen_address())
__all__.append("wcslen")


_lib.get_wcswcs_address.argtypes = ()
_lib.get_wcswcs_address.restype = ctypes.c_void_p
with _target_info:
    _wcswcs_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, const wchar_t *) noexcept(true)")
wcswcs = _wcswcs_signature.toctypes()(_lib.get_wcswcs_address())
__all__.append("wcswcs")


_lib.get_wcsnlen_address.argtypes = ()
_lib.get_wcsnlen_address.restype = ctypes.c_void_p
with _target_info:
    _wcsnlen_signature = rbc.typesystem.Type.fromstring("size_t (const wchar_t *, size_t) noexcept(true)")
wcsnlen = _wcsnlen_signature.toctypes()(_lib.get_wcsnlen_address())
__all__.append("wcsnlen")


_lib.get_wmemchr_address.argtypes = ()
_lib.get_wmemchr_address.restype = ctypes.c_void_p
with _target_info:
    _wmemchr_signature = rbc.typesystem.Type.fromstring("wchar_t *(const wchar_t *, wchar_t, size_t) noexcept(true)")
wmemchr = _wmemchr_signature.toctypes()(_lib.get_wmemchr_address())
__all__.append("wmemchr")


_lib.get_wmemcmp_address.argtypes = ()
_lib.get_wmemcmp_address.restype = ctypes.c_void_p
with _target_info:
    _wmemcmp_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, const wchar_t *, size_t) noexcept(true)")
wmemcmp = _wmemcmp_signature.toctypes()(_lib.get_wmemcmp_address())
__all__.append("wmemcmp")


_lib.get_wmemcpy_address.argtypes = ()
_lib.get_wmemcpy_address.restype = ctypes.c_void_p
with _target_info:
    _wmemcpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wmemcpy = _wmemcpy_signature.toctypes()(_lib.get_wmemcpy_address())
__all__.append("wmemcpy")


_lib.get_wmemmove_address.argtypes = ()
_lib.get_wmemmove_address.restype = ctypes.c_void_p
with _target_info:
    _wmemmove_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *, const wchar_t *, size_t) noexcept(true)")
wmemmove = _wmemmove_signature.toctypes()(_lib.get_wmemmove_address())
__all__.append("wmemmove")


_lib.get_wmemset_address.argtypes = ()
_lib.get_wmemset_address.restype = ctypes.c_void_p
with _target_info:
    _wmemset_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *, wchar_t, size_t) noexcept(true)")
wmemset = _wmemset_signature.toctypes()(_lib.get_wmemset_address())
__all__.append("wmemset")


_lib.get_wmempcpy_address.argtypes = ()
_lib.get_wmempcpy_address.restype = ctypes.c_void_p
with _target_info:
    _wmempcpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wmempcpy = _wmempcpy_signature.toctypes()(_lib.get_wmempcpy_address())
__all__.append("wmempcpy")


_lib.get_btowc_address.argtypes = ()
_lib.get_btowc_address.restype = ctypes.c_void_p
with _target_info:
    _btowc_signature = rbc.typesystem.Type.fromstring("wint_t (int) noexcept(true)")
btowc = _btowc_signature.toctypes()(_lib.get_btowc_address())
__all__.append("btowc")


_lib.get_wctob_address.argtypes = ()
_lib.get_wctob_address.restype = ctypes.c_void_p
with _target_info:
    _wctob_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
wctob = _wctob_signature.toctypes()(_lib.get_wctob_address())
__all__.append("wctob")


_lib.get_mbsinit_address.argtypes = ()
_lib.get_mbsinit_address.restype = ctypes.c_void_p
with _target_info:
    _mbsinit_signature = rbc.typesystem.Type.fromstring("int (const mbstate_t *) noexcept(true)")
mbsinit = _mbsinit_signature.toctypes()(_lib.get_mbsinit_address())
__all__.append("mbsinit")


_lib.get_mbrtowc_address.argtypes = ()
_lib.get_mbrtowc_address.restype = ctypes.c_void_p
with _target_info:
    _mbrtowc_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, const char *__restrict, size_t, mbstate_t *__restrict) noexcept(true)")
mbrtowc = _mbrtowc_signature.toctypes()(_lib.get_mbrtowc_address())
__all__.append("mbrtowc")


_lib.get_wcrtomb_address.argtypes = ()
_lib.get_wcrtomb_address.restype = ctypes.c_void_p
with _target_info:
    _wcrtomb_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, wchar_t, mbstate_t *__restrict) noexcept(true)")
wcrtomb = _wcrtomb_signature.toctypes()(_lib.get_wcrtomb_address())
__all__.append("wcrtomb")


_lib.get_mbrlen_address.argtypes = ()
_lib.get_mbrlen_address.restype = ctypes.c_void_p
with _target_info:
    _mbrlen_signature = rbc.typesystem.Type.fromstring("size_t (const char *__restrict, size_t, mbstate_t *__restrict) noexcept(true)")
mbrlen = _mbrlen_signature.toctypes()(_lib.get_mbrlen_address())
__all__.append("mbrlen")


_lib.get_mbsrtowcs_address.argtypes = ()
_lib.get_mbsrtowcs_address.restype = ctypes.c_void_p
with _target_info:
    _mbsrtowcs_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, const char **__restrict, size_t, mbstate_t *__restrict) noexcept(true)")
mbsrtowcs = _mbsrtowcs_signature.toctypes()(_lib.get_mbsrtowcs_address())
__all__.append("mbsrtowcs")


_lib.get_wcsrtombs_address.argtypes = ()
_lib.get_wcsrtombs_address.restype = ctypes.c_void_p
with _target_info:
    _wcsrtombs_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, const wchar_t **__restrict, size_t, mbstate_t *__restrict) noexcept(true)")
wcsrtombs = _wcsrtombs_signature.toctypes()(_lib.get_wcsrtombs_address())
__all__.append("wcsrtombs")


_lib.get_mbsnrtowcs_address.argtypes = ()
_lib.get_mbsnrtowcs_address.restype = ctypes.c_void_p
with _target_info:
    _mbsnrtowcs_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, const char **__restrict, size_t, size_t, mbstate_t *__restrict) noexcept(true)")
mbsnrtowcs = _mbsnrtowcs_signature.toctypes()(_lib.get_mbsnrtowcs_address())
__all__.append("mbsnrtowcs")


_lib.get_wcsnrtombs_address.argtypes = ()
_lib.get_wcsnrtombs_address.restype = ctypes.c_void_p
with _target_info:
    _wcsnrtombs_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, const wchar_t **__restrict, size_t, size_t, mbstate_t *__restrict) noexcept(true)")
wcsnrtombs = _wcsnrtombs_signature.toctypes()(_lib.get_wcsnrtombs_address())
__all__.append("wcsnrtombs")


_lib.get_wcwidth_address.argtypes = ()
_lib.get_wcwidth_address.restype = ctypes.c_void_p
with _target_info:
    _wcwidth_signature = rbc.typesystem.Type.fromstring("int (wchar_t) noexcept(true)")
wcwidth = _wcwidth_signature.toctypes()(_lib.get_wcwidth_address())
__all__.append("wcwidth")


_lib.get_wcswidth_address.argtypes = ()
_lib.get_wcswidth_address.restype = ctypes.c_void_p
with _target_info:
    _wcswidth_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *, size_t) noexcept(true)")
wcswidth = _wcswidth_signature.toctypes()(_lib.get_wcswidth_address())
__all__.append("wcswidth")


_lib.get_wcstod_address.argtypes = ()
_lib.get_wcstod_address.restype = ctypes.c_void_p
with _target_info:
    _wcstod_signature = rbc.typesystem.Type.fromstring("double (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstod = _wcstod_signature.toctypes()(_lib.get_wcstod_address())
__all__.append("wcstod")


_lib.get_wcstof_address.argtypes = ()
_lib.get_wcstof_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof_signature = rbc.typesystem.Type.fromstring("float (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstof = _wcstof_signature.toctypes()(_lib.get_wcstof_address())
__all__.append("wcstof")


_lib.get_wcstold_address.argtypes = ()
_lib.get_wcstold_address.restype = ctypes.c_void_p
with _target_info:
    _wcstold_signature = rbc.typesystem.Type.fromstring("long double (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstold = _wcstold_signature.toctypes()(_lib.get_wcstold_address())
__all__.append("wcstold")


_lib.get_wcstof32_address.argtypes = ()
_lib.get_wcstof32_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof32_signature = rbc.typesystem.Type.fromstring("_Float32 (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstof32 = _wcstof32_signature.toctypes()(_lib.get_wcstof32_address())
__all__.append("wcstof32")


_lib.get_wcstof64_address.argtypes = ()
_lib.get_wcstof64_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof64_signature = rbc.typesystem.Type.fromstring("_Float64 (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstof64 = _wcstof64_signature.toctypes()(_lib.get_wcstof64_address())
__all__.append("wcstof64")


_lib.get_wcstof32x_address.argtypes = ()
_lib.get_wcstof32x_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof32x_signature = rbc.typesystem.Type.fromstring("_Float32x (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstof32x = _wcstof32x_signature.toctypes()(_lib.get_wcstof32x_address())
__all__.append("wcstof32x")


_lib.get_wcstof64x_address.argtypes = ()
_lib.get_wcstof64x_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof64x_signature = rbc.typesystem.Type.fromstring("_Float64x (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true)")
wcstof64x = _wcstof64x_signature.toctypes()(_lib.get_wcstof64x_address())
__all__.append("wcstof64x")


_lib.get_wcstol_address.argtypes = ()
_lib.get_wcstol_address.restype = ctypes.c_void_p
with _target_info:
    _wcstol_signature = rbc.typesystem.Type.fromstring("long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstol = _wcstol_signature.toctypes()(_lib.get_wcstol_address())
__all__.append("wcstol")


_lib.get_wcstoul_address.argtypes = ()
_lib.get_wcstoul_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoul_signature = rbc.typesystem.Type.fromstring("unsigned long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstoul = _wcstoul_signature.toctypes()(_lib.get_wcstoul_address())
__all__.append("wcstoul")


_lib.get_wcstoll_address.argtypes = ()
_lib.get_wcstoll_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoll_signature = rbc.typesystem.Type.fromstring("long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstoll = _wcstoll_signature.toctypes()(_lib.get_wcstoll_address())
__all__.append("wcstoll")


_lib.get_wcstoull_address.argtypes = ()
_lib.get_wcstoull_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoull_signature = rbc.typesystem.Type.fromstring("unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstoull = _wcstoull_signature.toctypes()(_lib.get_wcstoull_address())
__all__.append("wcstoull")


_lib.get_wcstoq_address.argtypes = ()
_lib.get_wcstoq_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoq_signature = rbc.typesystem.Type.fromstring("long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstoq = _wcstoq_signature.toctypes()(_lib.get_wcstoq_address())
__all__.append("wcstoq")


_lib.get_wcstouq_address.argtypes = ()
_lib.get_wcstouq_address.restype = ctypes.c_void_p
with _target_info:
    _wcstouq_signature = rbc.typesystem.Type.fromstring("unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true)")
wcstouq = _wcstouq_signature.toctypes()(_lib.get_wcstouq_address())
__all__.append("wcstouq")


_lib.get_wcstol_l_address.argtypes = ()
_lib.get_wcstol_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstol_l_signature = rbc.typesystem.Type.fromstring("long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true)")
wcstol_l = _wcstol_l_signature.toctypes()(_lib.get_wcstol_l_address())
__all__.append("wcstol_l")


_lib.get_wcstoul_l_address.argtypes = ()
_lib.get_wcstoul_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoul_l_signature = rbc.typesystem.Type.fromstring("unsigned long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true)")
wcstoul_l = _wcstoul_l_signature.toctypes()(_lib.get_wcstoul_l_address())
__all__.append("wcstoul_l")


_lib.get_wcstoll_l_address.argtypes = ()
_lib.get_wcstoll_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoll_l_signature = rbc.typesystem.Type.fromstring("long long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true)")
wcstoll_l = _wcstoll_l_signature.toctypes()(_lib.get_wcstoll_l_address())
__all__.append("wcstoll_l")


_lib.get_wcstoull_l_address.argtypes = ()
_lib.get_wcstoull_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstoull_l_signature = rbc.typesystem.Type.fromstring("unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true)")
wcstoull_l = _wcstoull_l_signature.toctypes()(_lib.get_wcstoull_l_address())
__all__.append("wcstoull_l")


_lib.get_wcstod_l_address.argtypes = ()
_lib.get_wcstod_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstod_l_signature = rbc.typesystem.Type.fromstring("double (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstod_l = _wcstod_l_signature.toctypes()(_lib.get_wcstod_l_address())
__all__.append("wcstod_l")


_lib.get_wcstof_l_address.argtypes = ()
_lib.get_wcstof_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof_l_signature = rbc.typesystem.Type.fromstring("float (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstof_l = _wcstof_l_signature.toctypes()(_lib.get_wcstof_l_address())
__all__.append("wcstof_l")


_lib.get_wcstold_l_address.argtypes = ()
_lib.get_wcstold_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstold_l_signature = rbc.typesystem.Type.fromstring("long double (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstold_l = _wcstold_l_signature.toctypes()(_lib.get_wcstold_l_address())
__all__.append("wcstold_l")


_lib.get_wcstof32_l_address.argtypes = ()
_lib.get_wcstof32_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof32_l_signature = rbc.typesystem.Type.fromstring("_Float32 (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstof32_l = _wcstof32_l_signature.toctypes()(_lib.get_wcstof32_l_address())
__all__.append("wcstof32_l")


_lib.get_wcstof64_l_address.argtypes = ()
_lib.get_wcstof64_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof64_l_signature = rbc.typesystem.Type.fromstring("_Float64 (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstof64_l = _wcstof64_l_signature.toctypes()(_lib.get_wcstof64_l_address())
__all__.append("wcstof64_l")


_lib.get_wcstof32x_l_address.argtypes = ()
_lib.get_wcstof32x_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof32x_l_signature = rbc.typesystem.Type.fromstring("_Float32x (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstof32x_l = _wcstof32x_l_signature.toctypes()(_lib.get_wcstof32x_l_address())
__all__.append("wcstof32x_l")


_lib.get_wcstof64x_l_address.argtypes = ()
_lib.get_wcstof64x_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcstof64x_l_signature = rbc.typesystem.Type.fromstring("_Float64x (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true)")
wcstof64x_l = _wcstof64x_l_signature.toctypes()(_lib.get_wcstof64x_l_address())
__all__.append("wcstof64x_l")


_lib.get_wcpcpy_address.argtypes = ()
_lib.get_wcpcpy_address.restype = ctypes.c_void_p
with _target_info:
    _wcpcpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true)")
wcpcpy = _wcpcpy_signature.toctypes()(_lib.get_wcpcpy_address())
__all__.append("wcpcpy")


_lib.get_wcpncpy_address.argtypes = ()
_lib.get_wcpncpy_address.restype = ctypes.c_void_p
with _target_info:
    _wcpncpy_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wcpncpy = _wcpncpy_signature.toctypes()(_lib.get_wcpncpy_address())
__all__.append("wcpncpy")


_lib.get_open_wmemstream_address.argtypes = ()
_lib.get_open_wmemstream_address.restype = ctypes.c_void_p
with _target_info:
    _open_wmemstream_signature = rbc.typesystem.Type.fromstring("__FILE *(wchar_t **, size_t *) noexcept(true)")
open_wmemstream = _open_wmemstream_signature.toctypes()(_lib.get_open_wmemstream_address())
__all__.append("open_wmemstream")


_lib.get_fwide_address.argtypes = ()
_lib.get_fwide_address.restype = ctypes.c_void_p
with _target_info:
    _fwide_signature = rbc.typesystem.Type.fromstring("int (__FILE *, int) noexcept(true)")
fwide = _fwide_signature.toctypes()(_lib.get_fwide_address())
__all__.append("fwide")


_lib.get_fwprintf_address.argtypes = ()
_lib.get_fwprintf_address.restype = ctypes.c_void_p
with _target_info:
    _fwprintf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, ...)")
fwprintf = _fwprintf_signature.toctypes()(_lib.get_fwprintf_address())
__all__.append("fwprintf")


_lib.get_wprintf_address.argtypes = ()
_lib.get_wprintf_address.restype = ctypes.c_void_p
with _target_info:
    _wprintf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, ...)")
wprintf = _wprintf_signature.toctypes()(_lib.get_wprintf_address())
__all__.append("wprintf")


_lib.get_swprintf_address.argtypes = ()
_lib.get_swprintf_address.restype = ctypes.c_void_p
with _target_info:
    _swprintf_signature = rbc.typesystem.Type.fromstring("int (wchar_t *__restrict, size_t, const wchar_t *__restrict, ...) noexcept(true)")
swprintf = _swprintf_signature.toctypes()(_lib.get_swprintf_address())
__all__.append("swprintf")


_lib.get_vfwprintf_address.argtypes = ()
_lib.get_vfwprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vfwprintf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *)")
vfwprintf = _vfwprintf_signature.toctypes()(_lib.get_vfwprintf_address())
__all__.append("vfwprintf")


_lib.get_vwprintf_address.argtypes = ()
_lib.get_vwprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vwprintf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, __va_list_tag *)")
vwprintf = _vwprintf_signature.toctypes()(_lib.get_vwprintf_address())
__all__.append("vwprintf")


_lib.get_vswprintf_address.argtypes = ()
_lib.get_vswprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vswprintf_signature = rbc.typesystem.Type.fromstring("int (wchar_t *__restrict, size_t, const wchar_t *__restrict, __va_list_tag *) noexcept(true)")
vswprintf = _vswprintf_signature.toctypes()(_lib.get_vswprintf_address())
__all__.append("vswprintf")


_lib.get_fwscanf_address.argtypes = ()
_lib.get_fwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _fwscanf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, ...)")
fwscanf = _fwscanf_signature.toctypes()(_lib.get_fwscanf_address())
__all__.append("fwscanf")


_lib.get_wscanf_address.argtypes = ()
_lib.get_wscanf_address.restype = ctypes.c_void_p
with _target_info:
    _wscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, ...)")
wscanf = _wscanf_signature.toctypes()(_lib.get_wscanf_address())
__all__.append("wscanf")


_lib.get_swscanf_address.argtypes = ()
_lib.get_swscanf_address.restype = ctypes.c_void_p
with _target_info:
    _swscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, const wchar_t *__restrict, ...) noexcept(true)")
swscanf = _swscanf_signature.toctypes()(_lib.get_swscanf_address())
__all__.append("swscanf")


_lib.get_fwscanf_address.argtypes = ()
_lib.get_fwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _fwscanf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, ...)")
fwscanf = _fwscanf_signature.toctypes()(_lib.get_fwscanf_address())
__all__.append("fwscanf")


_lib.get_wscanf_address.argtypes = ()
_lib.get_wscanf_address.restype = ctypes.c_void_p
with _target_info:
    _wscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, ...)")
wscanf = _wscanf_signature.toctypes()(_lib.get_wscanf_address())
__all__.append("wscanf")


_lib.get_swscanf_address.argtypes = ()
_lib.get_swscanf_address.restype = ctypes.c_void_p
with _target_info:
    _swscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, const wchar_t *__restrict, ...) noexcept(true)")
swscanf = _swscanf_signature.toctypes()(_lib.get_swscanf_address())
__all__.append("swscanf")


_lib.get_vfwscanf_address.argtypes = ()
_lib.get_vfwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vfwscanf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *)")
vfwscanf = _vfwscanf_signature.toctypes()(_lib.get_vfwscanf_address())
__all__.append("vfwscanf")


_lib.get_vwscanf_address.argtypes = ()
_lib.get_vwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vwscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, __va_list_tag *)")
vwscanf = _vwscanf_signature.toctypes()(_lib.get_vwscanf_address())
__all__.append("vwscanf")


_lib.get_vswscanf_address.argtypes = ()
_lib.get_vswscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vswscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, const wchar_t *__restrict, __va_list_tag *) noexcept(true)")
vswscanf = _vswscanf_signature.toctypes()(_lib.get_vswscanf_address())
__all__.append("vswscanf")


_lib.get_vfwscanf_address.argtypes = ()
_lib.get_vfwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vfwscanf_signature = rbc.typesystem.Type.fromstring("int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *)")
vfwscanf = _vfwscanf_signature.toctypes()(_lib.get_vfwscanf_address())
__all__.append("vfwscanf")


_lib.get_vwscanf_address.argtypes = ()
_lib.get_vwscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vwscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, __va_list_tag *)")
vwscanf = _vwscanf_signature.toctypes()(_lib.get_vwscanf_address())
__all__.append("vwscanf")


_lib.get_vswscanf_address.argtypes = ()
_lib.get_vswscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vswscanf_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, const wchar_t *__restrict, __va_list_tag *) noexcept(true)")
vswscanf = _vswscanf_signature.toctypes()(_lib.get_vswscanf_address())
__all__.append("vswscanf")


_lib.get_fgetwc_address.argtypes = ()
_lib.get_fgetwc_address.restype = ctypes.c_void_p
with _target_info:
    _fgetwc_signature = rbc.typesystem.Type.fromstring("wint_t (__FILE *)")
fgetwc = _fgetwc_signature.toctypes()(_lib.get_fgetwc_address())
__all__.append("fgetwc")


_lib.get_getwc_address.argtypes = ()
_lib.get_getwc_address.restype = ctypes.c_void_p
with _target_info:
    _getwc_signature = rbc.typesystem.Type.fromstring("wint_t (__FILE *)")
getwc = _getwc_signature.toctypes()(_lib.get_getwc_address())
__all__.append("getwc")


_lib.get_getwchar_address.argtypes = ()
_lib.get_getwchar_address.restype = ctypes.c_void_p
with _target_info:
    _getwchar_signature = rbc.typesystem.Type.fromstring("wint_t ()")
getwchar = _getwchar_signature.toctypes()(_lib.get_getwchar_address())
__all__.append("getwchar")


_lib.get_fputwc_address.argtypes = ()
_lib.get_fputwc_address.restype = ctypes.c_void_p
with _target_info:
    _fputwc_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t, __FILE *)")
fputwc = _fputwc_signature.toctypes()(_lib.get_fputwc_address())
__all__.append("fputwc")


_lib.get_putwc_address.argtypes = ()
_lib.get_putwc_address.restype = ctypes.c_void_p
with _target_info:
    _putwc_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t, __FILE *)")
putwc = _putwc_signature.toctypes()(_lib.get_putwc_address())
__all__.append("putwc")


_lib.get_putwchar_address.argtypes = ()
_lib.get_putwchar_address.restype = ctypes.c_void_p
with _target_info:
    _putwchar_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t)")
putwchar = _putwchar_signature.toctypes()(_lib.get_putwchar_address())
__all__.append("putwchar")


_lib.get_fgetws_address.argtypes = ()
_lib.get_fgetws_address.restype = ctypes.c_void_p
with _target_info:
    _fgetws_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, int, __FILE *__restrict)")
fgetws = _fgetws_signature.toctypes()(_lib.get_fgetws_address())
__all__.append("fgetws")


_lib.get_fputws_address.argtypes = ()
_lib.get_fputws_address.restype = ctypes.c_void_p
with _target_info:
    _fputws_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, __FILE *__restrict)")
fputws = _fputws_signature.toctypes()(_lib.get_fputws_address())
__all__.append("fputws")


_lib.get_ungetwc_address.argtypes = ()
_lib.get_ungetwc_address.restype = ctypes.c_void_p
with _target_info:
    _ungetwc_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t, __FILE *)")
ungetwc = _ungetwc_signature.toctypes()(_lib.get_ungetwc_address())
__all__.append("ungetwc")


_lib.get_getwc_unlocked_address.argtypes = ()
_lib.get_getwc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _getwc_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t (__FILE *)")
getwc_unlocked = _getwc_unlocked_signature.toctypes()(_lib.get_getwc_unlocked_address())
__all__.append("getwc_unlocked")


_lib.get_getwchar_unlocked_address.argtypes = ()
_lib.get_getwchar_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _getwchar_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t ()")
getwchar_unlocked = _getwchar_unlocked_signature.toctypes()(_lib.get_getwchar_unlocked_address())
__all__.append("getwchar_unlocked")


_lib.get_fgetwc_unlocked_address.argtypes = ()
_lib.get_fgetwc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fgetwc_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t (__FILE *)")
fgetwc_unlocked = _fgetwc_unlocked_signature.toctypes()(_lib.get_fgetwc_unlocked_address())
__all__.append("fgetwc_unlocked")


_lib.get_fputwc_unlocked_address.argtypes = ()
_lib.get_fputwc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fputwc_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t, __FILE *)")
fputwc_unlocked = _fputwc_unlocked_signature.toctypes()(_lib.get_fputwc_unlocked_address())
__all__.append("fputwc_unlocked")


_lib.get_putwc_unlocked_address.argtypes = ()
_lib.get_putwc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _putwc_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t, __FILE *)")
putwc_unlocked = _putwc_unlocked_signature.toctypes()(_lib.get_putwc_unlocked_address())
__all__.append("putwc_unlocked")


_lib.get_putwchar_unlocked_address.argtypes = ()
_lib.get_putwchar_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _putwchar_unlocked_signature = rbc.typesystem.Type.fromstring("wint_t (wchar_t)")
putwchar_unlocked = _putwchar_unlocked_signature.toctypes()(_lib.get_putwchar_unlocked_address())
__all__.append("putwchar_unlocked")


_lib.get_fgetws_unlocked_address.argtypes = ()
_lib.get_fgetws_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fgetws_unlocked_signature = rbc.typesystem.Type.fromstring("wchar_t *(wchar_t *__restrict, int, __FILE *__restrict)")
fgetws_unlocked = _fgetws_unlocked_signature.toctypes()(_lib.get_fgetws_unlocked_address())
__all__.append("fgetws_unlocked")


_lib.get_fputws_unlocked_address.argtypes = ()
_lib.get_fputws_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fputws_unlocked_signature = rbc.typesystem.Type.fromstring("int (const wchar_t *__restrict, __FILE *__restrict)")
fputws_unlocked = _fputws_unlocked_signature.toctypes()(_lib.get_fputws_unlocked_address())
__all__.append("fputws_unlocked")


_lib.get_wcsftime_address.argtypes = ()
_lib.get_wcsftime_address.restype = ctypes.c_void_p
with _target_info:
    _wcsftime_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, size_t, const wchar_t *__restrict, const struct tm *__restrict) noexcept(true)")
wcsftime = _wcsftime_signature.toctypes()(_lib.get_wcsftime_address())
__all__.append("wcsftime")


_lib.get_wcsftime_l_address.argtypes = ()
_lib.get_wcsftime_l_address.restype = ctypes.c_void_p
with _target_info:
    _wcsftime_l_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, size_t, const wchar_t *__restrict, const struct tm *__restrict, locale_t) noexcept(true)")
wcsftime_l = _wcsftime_l_signature.toctypes()(_lib.get_wcsftime_l_address())
__all__.append("wcsftime_l")


_lib.get_setlocale_address.argtypes = ()
_lib.get_setlocale_address.restype = ctypes.c_void_p
with _target_info:
    _setlocale_signature = rbc.typesystem.Type.fromstring("char *(int, const char *) noexcept(true)")
setlocale = _setlocale_signature.toctypes()(_lib.get_setlocale_address())
__all__.append("setlocale")


_lib.get_localeconv_address.argtypes = ()
_lib.get_localeconv_address.restype = ctypes.c_void_p
with _target_info:
    _localeconv_signature = rbc.typesystem.Type.fromstring("struct lconv *() noexcept(true)")
localeconv = _localeconv_signature.toctypes()(_lib.get_localeconv_address())
__all__.append("localeconv")


_lib.get_newlocale_address.argtypes = ()
_lib.get_newlocale_address.restype = ctypes.c_void_p
with _target_info:
    _newlocale_signature = rbc.typesystem.Type.fromstring("locale_t (int, const char *, locale_t) noexcept(true)")
newlocale = _newlocale_signature.toctypes()(_lib.get_newlocale_address())
__all__.append("newlocale")


_lib.get_duplocale_address.argtypes = ()
_lib.get_duplocale_address.restype = ctypes.c_void_p
with _target_info:
    _duplocale_signature = rbc.typesystem.Type.fromstring("locale_t (locale_t) noexcept(true)")
duplocale = _duplocale_signature.toctypes()(_lib.get_duplocale_address())
__all__.append("duplocale")


_lib.get_freelocale_address.argtypes = ()
_lib.get_freelocale_address.restype = ctypes.c_void_p
with _target_info:
    _freelocale_signature = rbc.typesystem.Type.fromstring("void (locale_t) noexcept(true)")
freelocale = _freelocale_signature.toctypes()(_lib.get_freelocale_address())
__all__.append("freelocale")


_lib.get_uselocale_address.argtypes = ()
_lib.get_uselocale_address.restype = ctypes.c_void_p
with _target_info:
    _uselocale_signature = rbc.typesystem.Type.fromstring("locale_t (locale_t) noexcept(true)")
uselocale = _uselocale_signature.toctypes()(_lib.get_uselocale_address())
__all__.append("uselocale")


_lib.get_isalnum_address.argtypes = ()
_lib.get_isalnum_address.restype = ctypes.c_void_p
with _target_info:
    _isalnum_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isalnum = _isalnum_signature.toctypes()(_lib.get_isalnum_address())
__all__.append("isalnum")


_lib.get_isalpha_address.argtypes = ()
_lib.get_isalpha_address.restype = ctypes.c_void_p
with _target_info:
    _isalpha_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isalpha = _isalpha_signature.toctypes()(_lib.get_isalpha_address())
__all__.append("isalpha")


_lib.get_iscntrl_address.argtypes = ()
_lib.get_iscntrl_address.restype = ctypes.c_void_p
with _target_info:
    _iscntrl_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
iscntrl = _iscntrl_signature.toctypes()(_lib.get_iscntrl_address())
__all__.append("iscntrl")


_lib.get_isdigit_address.argtypes = ()
_lib.get_isdigit_address.restype = ctypes.c_void_p
with _target_info:
    _isdigit_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isdigit = _isdigit_signature.toctypes()(_lib.get_isdigit_address())
__all__.append("isdigit")


_lib.get_islower_address.argtypes = ()
_lib.get_islower_address.restype = ctypes.c_void_p
with _target_info:
    _islower_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
islower = _islower_signature.toctypes()(_lib.get_islower_address())
__all__.append("islower")


_lib.get_isgraph_address.argtypes = ()
_lib.get_isgraph_address.restype = ctypes.c_void_p
with _target_info:
    _isgraph_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isgraph = _isgraph_signature.toctypes()(_lib.get_isgraph_address())
__all__.append("isgraph")


_lib.get_isprint_address.argtypes = ()
_lib.get_isprint_address.restype = ctypes.c_void_p
with _target_info:
    _isprint_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isprint = _isprint_signature.toctypes()(_lib.get_isprint_address())
__all__.append("isprint")


_lib.get_ispunct_address.argtypes = ()
_lib.get_ispunct_address.restype = ctypes.c_void_p
with _target_info:
    _ispunct_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
ispunct = _ispunct_signature.toctypes()(_lib.get_ispunct_address())
__all__.append("ispunct")


_lib.get_isspace_address.argtypes = ()
_lib.get_isspace_address.restype = ctypes.c_void_p
with _target_info:
    _isspace_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isspace = _isspace_signature.toctypes()(_lib.get_isspace_address())
__all__.append("isspace")


_lib.get_isupper_address.argtypes = ()
_lib.get_isupper_address.restype = ctypes.c_void_p
with _target_info:
    _isupper_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isupper = _isupper_signature.toctypes()(_lib.get_isupper_address())
__all__.append("isupper")


_lib.get_isxdigit_address.argtypes = ()
_lib.get_isxdigit_address.restype = ctypes.c_void_p
with _target_info:
    _isxdigit_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isxdigit = _isxdigit_signature.toctypes()(_lib.get_isxdigit_address())
__all__.append("isxdigit")


_lib.get_tolower_address.argtypes = ()
_lib.get_tolower_address.restype = ctypes.c_void_p
with _target_info:
    _tolower_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
tolower = _tolower_signature.toctypes()(_lib.get_tolower_address())
__all__.append("tolower")


_lib.get_toupper_address.argtypes = ()
_lib.get_toupper_address.restype = ctypes.c_void_p
with _target_info:
    _toupper_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
toupper = _toupper_signature.toctypes()(_lib.get_toupper_address())
__all__.append("toupper")


_lib.get_isblank_address.argtypes = ()
_lib.get_isblank_address.restype = ctypes.c_void_p
with _target_info:
    _isblank_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isblank = _isblank_signature.toctypes()(_lib.get_isblank_address())
__all__.append("isblank")


_lib.get_isctype_address.argtypes = ()
_lib.get_isctype_address.restype = ctypes.c_void_p
with _target_info:
    _isctype_signature = rbc.typesystem.Type.fromstring("int (int, int) noexcept(true)")
isctype = _isctype_signature.toctypes()(_lib.get_isctype_address())
__all__.append("isctype")


_lib.get_isascii_address.argtypes = ()
_lib.get_isascii_address.restype = ctypes.c_void_p
with _target_info:
    _isascii_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
isascii = _isascii_signature.toctypes()(_lib.get_isascii_address())
__all__.append("isascii")


_lib.get_toascii_address.argtypes = ()
_lib.get_toascii_address.restype = ctypes.c_void_p
with _target_info:
    _toascii_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
toascii = _toascii_signature.toctypes()(_lib.get_toascii_address())
__all__.append("toascii")


_lib.get_isalnum_l_address.argtypes = ()
_lib.get_isalnum_l_address.restype = ctypes.c_void_p
with _target_info:
    _isalnum_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isalnum_l = _isalnum_l_signature.toctypes()(_lib.get_isalnum_l_address())
__all__.append("isalnum_l")


_lib.get_isalpha_l_address.argtypes = ()
_lib.get_isalpha_l_address.restype = ctypes.c_void_p
with _target_info:
    _isalpha_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isalpha_l = _isalpha_l_signature.toctypes()(_lib.get_isalpha_l_address())
__all__.append("isalpha_l")


_lib.get_iscntrl_l_address.argtypes = ()
_lib.get_iscntrl_l_address.restype = ctypes.c_void_p
with _target_info:
    _iscntrl_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
iscntrl_l = _iscntrl_l_signature.toctypes()(_lib.get_iscntrl_l_address())
__all__.append("iscntrl_l")


_lib.get_isdigit_l_address.argtypes = ()
_lib.get_isdigit_l_address.restype = ctypes.c_void_p
with _target_info:
    _isdigit_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isdigit_l = _isdigit_l_signature.toctypes()(_lib.get_isdigit_l_address())
__all__.append("isdigit_l")


_lib.get_islower_l_address.argtypes = ()
_lib.get_islower_l_address.restype = ctypes.c_void_p
with _target_info:
    _islower_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
islower_l = _islower_l_signature.toctypes()(_lib.get_islower_l_address())
__all__.append("islower_l")


_lib.get_isgraph_l_address.argtypes = ()
_lib.get_isgraph_l_address.restype = ctypes.c_void_p
with _target_info:
    _isgraph_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isgraph_l = _isgraph_l_signature.toctypes()(_lib.get_isgraph_l_address())
__all__.append("isgraph_l")


_lib.get_isprint_l_address.argtypes = ()
_lib.get_isprint_l_address.restype = ctypes.c_void_p
with _target_info:
    _isprint_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isprint_l = _isprint_l_signature.toctypes()(_lib.get_isprint_l_address())
__all__.append("isprint_l")


_lib.get_ispunct_l_address.argtypes = ()
_lib.get_ispunct_l_address.restype = ctypes.c_void_p
with _target_info:
    _ispunct_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
ispunct_l = _ispunct_l_signature.toctypes()(_lib.get_ispunct_l_address())
__all__.append("ispunct_l")


_lib.get_isspace_l_address.argtypes = ()
_lib.get_isspace_l_address.restype = ctypes.c_void_p
with _target_info:
    _isspace_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isspace_l = _isspace_l_signature.toctypes()(_lib.get_isspace_l_address())
__all__.append("isspace_l")


_lib.get_isupper_l_address.argtypes = ()
_lib.get_isupper_l_address.restype = ctypes.c_void_p
with _target_info:
    _isupper_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isupper_l = _isupper_l_signature.toctypes()(_lib.get_isupper_l_address())
__all__.append("isupper_l")


_lib.get_isxdigit_l_address.argtypes = ()
_lib.get_isxdigit_l_address.restype = ctypes.c_void_p
with _target_info:
    _isxdigit_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isxdigit_l = _isxdigit_l_signature.toctypes()(_lib.get_isxdigit_l_address())
__all__.append("isxdigit_l")


_lib.get_isblank_l_address.argtypes = ()
_lib.get_isblank_l_address.restype = ctypes.c_void_p
with _target_info:
    _isblank_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
isblank_l = _isblank_l_signature.toctypes()(_lib.get_isblank_l_address())
__all__.append("isblank_l")


_lib.get_tolower_l_address.argtypes = ()
_lib.get_tolower_l_address.restype = ctypes.c_void_p
with _target_info:
    _tolower_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
tolower_l = _tolower_l_signature.toctypes()(_lib.get_tolower_l_address())
__all__.append("tolower_l")


_lib.get_toupper_l_address.argtypes = ()
_lib.get_toupper_l_address.restype = ctypes.c_void_p
with _target_info:
    _toupper_l_signature = rbc.typesystem.Type.fromstring("int (int, locale_t) noexcept(true)")
toupper_l = _toupper_l_signature.toctypes()(_lib.get_toupper_l_address())
__all__.append("toupper_l")


_lib.get_clone_address.argtypes = ()
_lib.get_clone_address.restype = ctypes.c_void_p
with _target_info:
    _clone_signature = rbc.typesystem.Type.fromstring("int (int (*)(void *), void *, int, void *, ...) noexcept(true)")
clone = _clone_signature.toctypes()(_lib.get_clone_address())
__all__.append("clone")


_lib.get_unshare_address.argtypes = ()
_lib.get_unshare_address.restype = ctypes.c_void_p
with _target_info:
    _unshare_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
unshare = _unshare_signature.toctypes()(_lib.get_unshare_address())
__all__.append("unshare")


_lib.get_sched_getcpu_address.argtypes = ()
_lib.get_sched_getcpu_address.restype = ctypes.c_void_p
with _target_info:
    _sched_getcpu_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
sched_getcpu = _sched_getcpu_signature.toctypes()(_lib.get_sched_getcpu_address())
__all__.append("sched_getcpu")


_lib.get_getcpu_address.argtypes = ()
_lib.get_getcpu_address.restype = ctypes.c_void_p
with _target_info:
    _getcpu_signature = rbc.typesystem.Type.fromstring("int (unsigned int *, unsigned int *) noexcept(true)")
getcpu = _getcpu_signature.toctypes()(_lib.get_getcpu_address())
__all__.append("getcpu")


_lib.get_setns_address.argtypes = ()
_lib.get_setns_address.restype = ctypes.c_void_p
with _target_info:
    _setns_signature = rbc.typesystem.Type.fromstring("int (int, int) noexcept(true)")
setns = _setns_signature.toctypes()(_lib.get_setns_address())
__all__.append("setns")


_lib.get_sched_setparam_address.argtypes = ()
_lib.get_sched_setparam_address.restype = ctypes.c_void_p
with _target_info:
    _sched_setparam_signature = rbc.typesystem.Type.fromstring("int (__pid_t, const struct sched_param *) noexcept(true)")
sched_setparam = _sched_setparam_signature.toctypes()(_lib.get_sched_setparam_address())
__all__.append("sched_setparam")


_lib.get_sched_getparam_address.argtypes = ()
_lib.get_sched_getparam_address.restype = ctypes.c_void_p
with _target_info:
    _sched_getparam_signature = rbc.typesystem.Type.fromstring("int (__pid_t, struct sched_param *) noexcept(true)")
sched_getparam = _sched_getparam_signature.toctypes()(_lib.get_sched_getparam_address())
__all__.append("sched_getparam")


_lib.get_sched_setscheduler_address.argtypes = ()
_lib.get_sched_setscheduler_address.restype = ctypes.c_void_p
with _target_info:
    _sched_setscheduler_signature = rbc.typesystem.Type.fromstring("int (__pid_t, int, const struct sched_param *) noexcept(true)")
sched_setscheduler = _sched_setscheduler_signature.toctypes()(_lib.get_sched_setscheduler_address())
__all__.append("sched_setscheduler")


_lib.get_sched_getscheduler_address.argtypes = ()
_lib.get_sched_getscheduler_address.restype = ctypes.c_void_p
with _target_info:
    _sched_getscheduler_signature = rbc.typesystem.Type.fromstring("int (__pid_t) noexcept(true)")
sched_getscheduler = _sched_getscheduler_signature.toctypes()(_lib.get_sched_getscheduler_address())
__all__.append("sched_getscheduler")


_lib.get_sched_yield_address.argtypes = ()
_lib.get_sched_yield_address.restype = ctypes.c_void_p
with _target_info:
    _sched_yield_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
sched_yield = _sched_yield_signature.toctypes()(_lib.get_sched_yield_address())
__all__.append("sched_yield")


_lib.get_sched_get_priority_max_address.argtypes = ()
_lib.get_sched_get_priority_max_address.restype = ctypes.c_void_p
with _target_info:
    _sched_get_priority_max_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
sched_get_priority_max = _sched_get_priority_max_signature.toctypes()(_lib.get_sched_get_priority_max_address())
__all__.append("sched_get_priority_max")


_lib.get_sched_get_priority_min_address.argtypes = ()
_lib.get_sched_get_priority_min_address.restype = ctypes.c_void_p
with _target_info:
    _sched_get_priority_min_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
sched_get_priority_min = _sched_get_priority_min_signature.toctypes()(_lib.get_sched_get_priority_min_address())
__all__.append("sched_get_priority_min")


_lib.get_sched_rr_get_interval_address.argtypes = ()
_lib.get_sched_rr_get_interval_address.restype = ctypes.c_void_p
with _target_info:
    _sched_rr_get_interval_signature = rbc.typesystem.Type.fromstring("int (__pid_t, struct timespec *) noexcept(true)")
sched_rr_get_interval = _sched_rr_get_interval_signature.toctypes()(_lib.get_sched_rr_get_interval_address())
__all__.append("sched_rr_get_interval")


_lib.get_sched_setaffinity_address.argtypes = ()
_lib.get_sched_setaffinity_address.restype = ctypes.c_void_p
with _target_info:
    _sched_setaffinity_signature = rbc.typesystem.Type.fromstring("int (__pid_t, size_t, const cpu_set_t *) noexcept(true)")
sched_setaffinity = _sched_setaffinity_signature.toctypes()(_lib.get_sched_setaffinity_address())
__all__.append("sched_setaffinity")


_lib.get_sched_getaffinity_address.argtypes = ()
_lib.get_sched_getaffinity_address.restype = ctypes.c_void_p
with _target_info:
    _sched_getaffinity_signature = rbc.typesystem.Type.fromstring("int (__pid_t, size_t, cpu_set_t *) noexcept(true)")
sched_getaffinity = _sched_getaffinity_signature.toctypes()(_lib.get_sched_getaffinity_address())
__all__.append("sched_getaffinity")


_lib.get_clock_adjtime_address.argtypes = ()
_lib.get_clock_adjtime_address.restype = ctypes.c_void_p
with _target_info:
    _clock_adjtime_signature = rbc.typesystem.Type.fromstring("int (__clockid_t, struct timex *) noexcept(true)")
clock_adjtime = _clock_adjtime_signature.toctypes()(_lib.get_clock_adjtime_address())
__all__.append("clock_adjtime")


_lib.get_clock_address.argtypes = ()
_lib.get_clock_address.restype = ctypes.c_void_p
with _target_info:
    _clock_signature = rbc.typesystem.Type.fromstring("clock_t () noexcept(true)")
clock = _clock_signature.toctypes()(_lib.get_clock_address())
__all__.append("clock")


_lib.get_time_address.argtypes = ()
_lib.get_time_address.restype = ctypes.c_void_p
with _target_info:
    _time_signature = rbc.typesystem.Type.fromstring("time_t (time_t *) noexcept(true)")
time = _time_signature.toctypes()(_lib.get_time_address())
__all__.append("time")


_lib.get_difftime_address.argtypes = ()
_lib.get_difftime_address.restype = ctypes.c_void_p
with _target_info:
    _difftime_signature = rbc.typesystem.Type.fromstring("double (time_t, time_t) noexcept(true)")
difftime = _difftime_signature.toctypes()(_lib.get_difftime_address())
__all__.append("difftime")


_lib.get_mktime_address.argtypes = ()
_lib.get_mktime_address.restype = ctypes.c_void_p
with _target_info:
    _mktime_signature = rbc.typesystem.Type.fromstring("time_t (struct tm *) noexcept(true)")
mktime = _mktime_signature.toctypes()(_lib.get_mktime_address())
__all__.append("mktime")


_lib.get_strftime_address.argtypes = ()
_lib.get_strftime_address.restype = ctypes.c_void_p
with _target_info:
    _strftime_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, size_t, const char *__restrict, const struct tm *__restrict) noexcept(true)")
strftime = _strftime_signature.toctypes()(_lib.get_strftime_address())
__all__.append("strftime")


_lib.get_strptime_address.argtypes = ()
_lib.get_strptime_address.restype = ctypes.c_void_p
with _target_info:
    _strptime_signature = rbc.typesystem.Type.fromstring("char *(const char *__restrict, const char *__restrict, struct tm *) noexcept(true)")
strptime = _strptime_signature.toctypes()(_lib.get_strptime_address())
__all__.append("strptime")


_lib.get_strftime_l_address.argtypes = ()
_lib.get_strftime_l_address.restype = ctypes.c_void_p
with _target_info:
    _strftime_l_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, size_t, const char *__restrict, const struct tm *__restrict, locale_t) noexcept(true)")
strftime_l = _strftime_l_signature.toctypes()(_lib.get_strftime_l_address())
__all__.append("strftime_l")


_lib.get_strptime_l_address.argtypes = ()
_lib.get_strptime_l_address.restype = ctypes.c_void_p
with _target_info:
    _strptime_l_signature = rbc.typesystem.Type.fromstring("char *(const char *__restrict, const char *__restrict, struct tm *, locale_t) noexcept(true)")
strptime_l = _strptime_l_signature.toctypes()(_lib.get_strptime_l_address())
__all__.append("strptime_l")


_lib.get_gmtime_address.argtypes = ()
_lib.get_gmtime_address.restype = ctypes.c_void_p
with _target_info:
    _gmtime_signature = rbc.typesystem.Type.fromstring("struct tm *(const time_t *) noexcept(true)")
gmtime = _gmtime_signature.toctypes()(_lib.get_gmtime_address())
__all__.append("gmtime")


_lib.get_localtime_address.argtypes = ()
_lib.get_localtime_address.restype = ctypes.c_void_p
with _target_info:
    _localtime_signature = rbc.typesystem.Type.fromstring("struct tm *(const time_t *) noexcept(true)")
localtime = _localtime_signature.toctypes()(_lib.get_localtime_address())
__all__.append("localtime")


_lib.get_gmtime_r_address.argtypes = ()
_lib.get_gmtime_r_address.restype = ctypes.c_void_p
with _target_info:
    _gmtime_r_signature = rbc.typesystem.Type.fromstring("struct tm *(const time_t *__restrict, struct tm *__restrict) noexcept(true)")
gmtime_r = _gmtime_r_signature.toctypes()(_lib.get_gmtime_r_address())
__all__.append("gmtime_r")


_lib.get_localtime_r_address.argtypes = ()
_lib.get_localtime_r_address.restype = ctypes.c_void_p
with _target_info:
    _localtime_r_signature = rbc.typesystem.Type.fromstring("struct tm *(const time_t *__restrict, struct tm *__restrict) noexcept(true)")
localtime_r = _localtime_r_signature.toctypes()(_lib.get_localtime_r_address())
__all__.append("localtime_r")


_lib.get_asctime_address.argtypes = ()
_lib.get_asctime_address.restype = ctypes.c_void_p
with _target_info:
    _asctime_signature = rbc.typesystem.Type.fromstring("char *(const struct tm *) noexcept(true)")
asctime = _asctime_signature.toctypes()(_lib.get_asctime_address())
__all__.append("asctime")


_lib.get_ctime_address.argtypes = ()
_lib.get_ctime_address.restype = ctypes.c_void_p
with _target_info:
    _ctime_signature = rbc.typesystem.Type.fromstring("char *(const time_t *) noexcept(true)")
ctime = _ctime_signature.toctypes()(_lib.get_ctime_address())
__all__.append("ctime")


_lib.get_asctime_r_address.argtypes = ()
_lib.get_asctime_r_address.restype = ctypes.c_void_p
with _target_info:
    _asctime_r_signature = rbc.typesystem.Type.fromstring("char *(const struct tm *__restrict, char *__restrict) noexcept(true)")
asctime_r = _asctime_r_signature.toctypes()(_lib.get_asctime_r_address())
__all__.append("asctime_r")


_lib.get_ctime_r_address.argtypes = ()
_lib.get_ctime_r_address.restype = ctypes.c_void_p
with _target_info:
    _ctime_r_signature = rbc.typesystem.Type.fromstring("char *(const time_t *__restrict, char *__restrict) noexcept(true)")
ctime_r = _ctime_r_signature.toctypes()(_lib.get_ctime_r_address())
__all__.append("ctime_r")


_lib.get_tzset_address.argtypes = ()
_lib.get_tzset_address.restype = ctypes.c_void_p
with _target_info:
    _tzset_signature = rbc.typesystem.Type.fromstring("void () noexcept(true)")
tzset = _tzset_signature.toctypes()(_lib.get_tzset_address())
__all__.append("tzset")


_lib.get_timegm_address.argtypes = ()
_lib.get_timegm_address.restype = ctypes.c_void_p
with _target_info:
    _timegm_signature = rbc.typesystem.Type.fromstring("time_t (struct tm *) noexcept(true)")
timegm = _timegm_signature.toctypes()(_lib.get_timegm_address())
__all__.append("timegm")


_lib.get_timelocal_address.argtypes = ()
_lib.get_timelocal_address.restype = ctypes.c_void_p
with _target_info:
    _timelocal_signature = rbc.typesystem.Type.fromstring("time_t (struct tm *) noexcept(true)")
timelocal = _timelocal_signature.toctypes()(_lib.get_timelocal_address())
__all__.append("timelocal")


_lib.get_dysize_address.argtypes = ()
_lib.get_dysize_address.restype = ctypes.c_void_p
with _target_info:
    _dysize_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
dysize = _dysize_signature.toctypes()(_lib.get_dysize_address())
__all__.append("dysize")


_lib.get_nanosleep_address.argtypes = ()
_lib.get_nanosleep_address.restype = ctypes.c_void_p
with _target_info:
    _nanosleep_signature = rbc.typesystem.Type.fromstring("int (const struct timespec *, struct timespec *)")
nanosleep = _nanosleep_signature.toctypes()(_lib.get_nanosleep_address())
__all__.append("nanosleep")


_lib.get_clock_getres_address.argtypes = ()
_lib.get_clock_getres_address.restype = ctypes.c_void_p
with _target_info:
    _clock_getres_signature = rbc.typesystem.Type.fromstring("int (clockid_t, struct timespec *) noexcept(true)")
clock_getres = _clock_getres_signature.toctypes()(_lib.get_clock_getres_address())
__all__.append("clock_getres")


_lib.get_clock_gettime_address.argtypes = ()
_lib.get_clock_gettime_address.restype = ctypes.c_void_p
with _target_info:
    _clock_gettime_signature = rbc.typesystem.Type.fromstring("int (clockid_t, struct timespec *) noexcept(true)")
clock_gettime = _clock_gettime_signature.toctypes()(_lib.get_clock_gettime_address())
__all__.append("clock_gettime")


_lib.get_clock_settime_address.argtypes = ()
_lib.get_clock_settime_address.restype = ctypes.c_void_p
with _target_info:
    _clock_settime_signature = rbc.typesystem.Type.fromstring("int (clockid_t, const struct timespec *) noexcept(true)")
clock_settime = _clock_settime_signature.toctypes()(_lib.get_clock_settime_address())
__all__.append("clock_settime")


_lib.get_clock_nanosleep_address.argtypes = ()
_lib.get_clock_nanosleep_address.restype = ctypes.c_void_p
with _target_info:
    _clock_nanosleep_signature = rbc.typesystem.Type.fromstring("int (clockid_t, int, const struct timespec *, struct timespec *)")
clock_nanosleep = _clock_nanosleep_signature.toctypes()(_lib.get_clock_nanosleep_address())
__all__.append("clock_nanosleep")


_lib.get_clock_getcpuclockid_address.argtypes = ()
_lib.get_clock_getcpuclockid_address.restype = ctypes.c_void_p
with _target_info:
    _clock_getcpuclockid_signature = rbc.typesystem.Type.fromstring("int (pid_t, clockid_t *) noexcept(true)")
clock_getcpuclockid = _clock_getcpuclockid_signature.toctypes()(_lib.get_clock_getcpuclockid_address())
__all__.append("clock_getcpuclockid")


_lib.get_timer_create_address.argtypes = ()
_lib.get_timer_create_address.restype = ctypes.c_void_p
with _target_info:
    _timer_create_signature = rbc.typesystem.Type.fromstring("int (clockid_t, struct sigevent *__restrict, timer_t *__restrict) noexcept(true)")
timer_create = _timer_create_signature.toctypes()(_lib.get_timer_create_address())
__all__.append("timer_create")


_lib.get_timer_delete_address.argtypes = ()
_lib.get_timer_delete_address.restype = ctypes.c_void_p
with _target_info:
    _timer_delete_signature = rbc.typesystem.Type.fromstring("int (timer_t) noexcept(true)")
timer_delete = _timer_delete_signature.toctypes()(_lib.get_timer_delete_address())
__all__.append("timer_delete")


_lib.get_timer_settime_address.argtypes = ()
_lib.get_timer_settime_address.restype = ctypes.c_void_p
with _target_info:
    _timer_settime_signature = rbc.typesystem.Type.fromstring("int (timer_t, int, const struct itimerspec *__restrict, struct itimerspec *__restrict) noexcept(true)")
timer_settime = _timer_settime_signature.toctypes()(_lib.get_timer_settime_address())
__all__.append("timer_settime")


_lib.get_timer_gettime_address.argtypes = ()
_lib.get_timer_gettime_address.restype = ctypes.c_void_p
with _target_info:
    _timer_gettime_signature = rbc.typesystem.Type.fromstring("int (timer_t, struct itimerspec *) noexcept(true)")
timer_gettime = _timer_gettime_signature.toctypes()(_lib.get_timer_gettime_address())
__all__.append("timer_gettime")


_lib.get_timer_getoverrun_address.argtypes = ()
_lib.get_timer_getoverrun_address.restype = ctypes.c_void_p
with _target_info:
    _timer_getoverrun_signature = rbc.typesystem.Type.fromstring("int (timer_t) noexcept(true)")
timer_getoverrun = _timer_getoverrun_signature.toctypes()(_lib.get_timer_getoverrun_address())
__all__.append("timer_getoverrun")


_lib.get_timespec_get_address.argtypes = ()
_lib.get_timespec_get_address.restype = ctypes.c_void_p
with _target_info:
    _timespec_get_signature = rbc.typesystem.Type.fromstring("int (struct timespec *, int) noexcept(true)")
timespec_get = _timespec_get_signature.toctypes()(_lib.get_timespec_get_address())
__all__.append("timespec_get")


_lib.get_timespec_getres_address.argtypes = ()
_lib.get_timespec_getres_address.restype = ctypes.c_void_p
with _target_info:
    _timespec_getres_signature = rbc.typesystem.Type.fromstring("int (struct timespec *, int) noexcept(true)")
timespec_getres = _timespec_getres_signature.toctypes()(_lib.get_timespec_getres_address())
__all__.append("timespec_getres")


_lib.get_getdate_address.argtypes = ()
_lib.get_getdate_address.restype = ctypes.c_void_p
with _target_info:
    _getdate_signature = rbc.typesystem.Type.fromstring("struct tm *(const char *)")
getdate = _getdate_signature.toctypes()(_lib.get_getdate_address())
__all__.append("getdate")


_lib.get_getdate_r_address.argtypes = ()
_lib.get_getdate_r_address.restype = ctypes.c_void_p
with _target_info:
    _getdate_r_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, struct tm *__restrict)")
getdate_r = _getdate_r_signature.toctypes()(_lib.get_getdate_r_address())
__all__.append("getdate_r")


_lib.get_pthread_create_address.argtypes = ()
_lib.get_pthread_create_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_create_signature = rbc.typesystem.Type.fromstring("int (pthread_t *__restrict, const pthread_attr_t *__restrict, void *(*)(void *), void *__restrict) noexcept(true)")
pthread_create = _pthread_create_signature.toctypes()(_lib.get_pthread_create_address())
__all__.append("pthread_create")


_lib.get_pthread_exit_address.argtypes = ()
_lib.get_pthread_exit_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_exit_signature = rbc.typesystem.Type.fromstring("void (void *) __attribute__((noreturn))")
pthread_exit = _pthread_exit_signature.toctypes()(_lib.get_pthread_exit_address())
__all__.append("pthread_exit")


_lib.get_pthread_join_address.argtypes = ()
_lib.get_pthread_join_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_join_signature = rbc.typesystem.Type.fromstring("int (pthread_t, void **)")
pthread_join = _pthread_join_signature.toctypes()(_lib.get_pthread_join_address())
__all__.append("pthread_join")


_lib.get_pthread_tryjoin_np_address.argtypes = ()
_lib.get_pthread_tryjoin_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_tryjoin_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, void **) noexcept(true)")
pthread_tryjoin_np = _pthread_tryjoin_np_signature.toctypes()(_lib.get_pthread_tryjoin_np_address())
__all__.append("pthread_tryjoin_np")


_lib.get_pthread_timedjoin_np_address.argtypes = ()
_lib.get_pthread_timedjoin_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_timedjoin_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, void **, const struct timespec *)")
pthread_timedjoin_np = _pthread_timedjoin_np_signature.toctypes()(_lib.get_pthread_timedjoin_np_address())
__all__.append("pthread_timedjoin_np")


_lib.get_pthread_clockjoin_np_address.argtypes = ()
_lib.get_pthread_clockjoin_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_clockjoin_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, void **, clockid_t, const struct timespec *)")
pthread_clockjoin_np = _pthread_clockjoin_np_signature.toctypes()(_lib.get_pthread_clockjoin_np_address())
__all__.append("pthread_clockjoin_np")


_lib.get_pthread_detach_address.argtypes = ()
_lib.get_pthread_detach_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_detach_signature = rbc.typesystem.Type.fromstring("int (pthread_t) noexcept(true)")
pthread_detach = _pthread_detach_signature.toctypes()(_lib.get_pthread_detach_address())
__all__.append("pthread_detach")


_lib.get_pthread_self_address.argtypes = ()
_lib.get_pthread_self_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_self_signature = rbc.typesystem.Type.fromstring("pthread_t () noexcept(true)")
pthread_self = _pthread_self_signature.toctypes()(_lib.get_pthread_self_address())
__all__.append("pthread_self")


_lib.get_pthread_equal_address.argtypes = ()
_lib.get_pthread_equal_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_equal_signature = rbc.typesystem.Type.fromstring("int (pthread_t, pthread_t) noexcept(true)")
pthread_equal = _pthread_equal_signature.toctypes()(_lib.get_pthread_equal_address())
__all__.append("pthread_equal")


_lib.get_pthread_attr_init_address.argtypes = ()
_lib.get_pthread_attr_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_init_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *) noexcept(true)")
pthread_attr_init = _pthread_attr_init_signature.toctypes()(_lib.get_pthread_attr_init_address())
__all__.append("pthread_attr_init")


_lib.get_pthread_attr_destroy_address.argtypes = ()
_lib.get_pthread_attr_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *) noexcept(true)")
pthread_attr_destroy = _pthread_attr_destroy_signature.toctypes()(_lib.get_pthread_attr_destroy_address())
__all__.append("pthread_attr_destroy")


_lib.get_pthread_attr_getdetachstate_address.argtypes = ()
_lib.get_pthread_attr_getdetachstate_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getdetachstate_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *, int *) noexcept(true)")
pthread_attr_getdetachstate = _pthread_attr_getdetachstate_signature.toctypes()(_lib.get_pthread_attr_getdetachstate_address())
__all__.append("pthread_attr_getdetachstate")


_lib.get_pthread_attr_setdetachstate_address.argtypes = ()
_lib.get_pthread_attr_setdetachstate_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setdetachstate_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, int) noexcept(true)")
pthread_attr_setdetachstate = _pthread_attr_setdetachstate_signature.toctypes()(_lib.get_pthread_attr_setdetachstate_address())
__all__.append("pthread_attr_setdetachstate")


_lib.get_pthread_attr_getguardsize_address.argtypes = ()
_lib.get_pthread_attr_getguardsize_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getguardsize_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *, size_t *) noexcept(true)")
pthread_attr_getguardsize = _pthread_attr_getguardsize_signature.toctypes()(_lib.get_pthread_attr_getguardsize_address())
__all__.append("pthread_attr_getguardsize")


_lib.get_pthread_attr_setguardsize_address.argtypes = ()
_lib.get_pthread_attr_setguardsize_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setguardsize_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, size_t) noexcept(true)")
pthread_attr_setguardsize = _pthread_attr_setguardsize_signature.toctypes()(_lib.get_pthread_attr_setguardsize_address())
__all__.append("pthread_attr_setguardsize")


_lib.get_pthread_attr_getschedparam_address.argtypes = ()
_lib.get_pthread_attr_getschedparam_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getschedparam_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, struct sched_param *__restrict) noexcept(true)")
pthread_attr_getschedparam = _pthread_attr_getschedparam_signature.toctypes()(_lib.get_pthread_attr_getschedparam_address())
__all__.append("pthread_attr_getschedparam")


_lib.get_pthread_attr_setschedparam_address.argtypes = ()
_lib.get_pthread_attr_setschedparam_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setschedparam_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *__restrict, const struct sched_param *__restrict) noexcept(true)")
pthread_attr_setschedparam = _pthread_attr_setschedparam_signature.toctypes()(_lib.get_pthread_attr_setschedparam_address())
__all__.append("pthread_attr_setschedparam")


_lib.get_pthread_attr_getschedpolicy_address.argtypes = ()
_lib.get_pthread_attr_getschedpolicy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getschedpolicy_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true)")
pthread_attr_getschedpolicy = _pthread_attr_getschedpolicy_signature.toctypes()(_lib.get_pthread_attr_getschedpolicy_address())
__all__.append("pthread_attr_getschedpolicy")


_lib.get_pthread_attr_setschedpolicy_address.argtypes = ()
_lib.get_pthread_attr_setschedpolicy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setschedpolicy_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, int) noexcept(true)")
pthread_attr_setschedpolicy = _pthread_attr_setschedpolicy_signature.toctypes()(_lib.get_pthread_attr_setschedpolicy_address())
__all__.append("pthread_attr_setschedpolicy")


_lib.get_pthread_attr_getinheritsched_address.argtypes = ()
_lib.get_pthread_attr_getinheritsched_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getinheritsched_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true)")
pthread_attr_getinheritsched = _pthread_attr_getinheritsched_signature.toctypes()(_lib.get_pthread_attr_getinheritsched_address())
__all__.append("pthread_attr_getinheritsched")


_lib.get_pthread_attr_setinheritsched_address.argtypes = ()
_lib.get_pthread_attr_setinheritsched_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setinheritsched_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, int) noexcept(true)")
pthread_attr_setinheritsched = _pthread_attr_setinheritsched_signature.toctypes()(_lib.get_pthread_attr_setinheritsched_address())
__all__.append("pthread_attr_setinheritsched")


_lib.get_pthread_attr_getscope_address.argtypes = ()
_lib.get_pthread_attr_getscope_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getscope_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true)")
pthread_attr_getscope = _pthread_attr_getscope_signature.toctypes()(_lib.get_pthread_attr_getscope_address())
__all__.append("pthread_attr_getscope")


_lib.get_pthread_attr_setscope_address.argtypes = ()
_lib.get_pthread_attr_setscope_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setscope_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, int) noexcept(true)")
pthread_attr_setscope = _pthread_attr_setscope_signature.toctypes()(_lib.get_pthread_attr_setscope_address())
__all__.append("pthread_attr_setscope")


_lib.get_pthread_attr_getstackaddr_address.argtypes = ()
_lib.get_pthread_attr_getstackaddr_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getstackaddr_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, void **__restrict) noexcept(true)")
pthread_attr_getstackaddr = _pthread_attr_getstackaddr_signature.toctypes()(_lib.get_pthread_attr_getstackaddr_address())
__all__.append("pthread_attr_getstackaddr")


_lib.get_pthread_attr_setstackaddr_address.argtypes = ()
_lib.get_pthread_attr_setstackaddr_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setstackaddr_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, void *) noexcept(true)")
pthread_attr_setstackaddr = _pthread_attr_setstackaddr_signature.toctypes()(_lib.get_pthread_attr_setstackaddr_address())
__all__.append("pthread_attr_setstackaddr")


_lib.get_pthread_attr_getstacksize_address.argtypes = ()
_lib.get_pthread_attr_getstacksize_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getstacksize_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, size_t *__restrict) noexcept(true)")
pthread_attr_getstacksize = _pthread_attr_getstacksize_signature.toctypes()(_lib.get_pthread_attr_getstacksize_address())
__all__.append("pthread_attr_getstacksize")


_lib.get_pthread_attr_setstacksize_address.argtypes = ()
_lib.get_pthread_attr_setstacksize_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setstacksize_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, size_t) noexcept(true)")
pthread_attr_setstacksize = _pthread_attr_setstacksize_signature.toctypes()(_lib.get_pthread_attr_setstacksize_address())
__all__.append("pthread_attr_setstacksize")


_lib.get_pthread_attr_getstack_address.argtypes = ()
_lib.get_pthread_attr_getstack_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getstack_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *__restrict, void **__restrict, size_t *__restrict) noexcept(true)")
pthread_attr_getstack = _pthread_attr_getstack_signature.toctypes()(_lib.get_pthread_attr_getstack_address())
__all__.append("pthread_attr_getstack")


_lib.get_pthread_attr_setstack_address.argtypes = ()
_lib.get_pthread_attr_setstack_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setstack_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, void *, size_t) noexcept(true)")
pthread_attr_setstack = _pthread_attr_setstack_signature.toctypes()(_lib.get_pthread_attr_setstack_address())
__all__.append("pthread_attr_setstack")


_lib.get_pthread_attr_setaffinity_np_address.argtypes = ()
_lib.get_pthread_attr_setaffinity_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setaffinity_np_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, size_t, const cpu_set_t *) noexcept(true)")
pthread_attr_setaffinity_np = _pthread_attr_setaffinity_np_signature.toctypes()(_lib.get_pthread_attr_setaffinity_np_address())
__all__.append("pthread_attr_setaffinity_np")


_lib.get_pthread_attr_getaffinity_np_address.argtypes = ()
_lib.get_pthread_attr_getaffinity_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getaffinity_np_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *, size_t, cpu_set_t *) noexcept(true)")
pthread_attr_getaffinity_np = _pthread_attr_getaffinity_np_signature.toctypes()(_lib.get_pthread_attr_getaffinity_np_address())
__all__.append("pthread_attr_getaffinity_np")


_lib.get_pthread_getattr_default_np_address.argtypes = ()
_lib.get_pthread_getattr_default_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getattr_default_np_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *) noexcept(true)")
pthread_getattr_default_np = _pthread_getattr_default_np_signature.toctypes()(_lib.get_pthread_getattr_default_np_address())
__all__.append("pthread_getattr_default_np")


_lib.get_pthread_attr_setsigmask_np_address.argtypes = ()
_lib.get_pthread_attr_setsigmask_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_setsigmask_np_signature = rbc.typesystem.Type.fromstring("int (pthread_attr_t *, const __sigset_t *)")
pthread_attr_setsigmask_np = _pthread_attr_setsigmask_np_signature.toctypes()(_lib.get_pthread_attr_setsigmask_np_address())
__all__.append("pthread_attr_setsigmask_np")


_lib.get_pthread_attr_getsigmask_np_address.argtypes = ()
_lib.get_pthread_attr_getsigmask_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_attr_getsigmask_np_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *, __sigset_t *)")
pthread_attr_getsigmask_np = _pthread_attr_getsigmask_np_signature.toctypes()(_lib.get_pthread_attr_getsigmask_np_address())
__all__.append("pthread_attr_getsigmask_np")


_lib.get_pthread_setattr_default_np_address.argtypes = ()
_lib.get_pthread_setattr_default_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setattr_default_np_signature = rbc.typesystem.Type.fromstring("int (const pthread_attr_t *) noexcept(true)")
pthread_setattr_default_np = _pthread_setattr_default_np_signature.toctypes()(_lib.get_pthread_setattr_default_np_address())
__all__.append("pthread_setattr_default_np")


_lib.get_pthread_getattr_np_address.argtypes = ()
_lib.get_pthread_getattr_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getattr_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, pthread_attr_t *) noexcept(true)")
pthread_getattr_np = _pthread_getattr_np_signature.toctypes()(_lib.get_pthread_getattr_np_address())
__all__.append("pthread_getattr_np")


_lib.get_pthread_setschedparam_address.argtypes = ()
_lib.get_pthread_setschedparam_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setschedparam_signature = rbc.typesystem.Type.fromstring("int (pthread_t, int, const struct sched_param *) noexcept(true)")
pthread_setschedparam = _pthread_setschedparam_signature.toctypes()(_lib.get_pthread_setschedparam_address())
__all__.append("pthread_setschedparam")


_lib.get_pthread_getschedparam_address.argtypes = ()
_lib.get_pthread_getschedparam_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getschedparam_signature = rbc.typesystem.Type.fromstring("int (pthread_t, int *__restrict, struct sched_param *__restrict) noexcept(true)")
pthread_getschedparam = _pthread_getschedparam_signature.toctypes()(_lib.get_pthread_getschedparam_address())
__all__.append("pthread_getschedparam")


_lib.get_pthread_setschedprio_address.argtypes = ()
_lib.get_pthread_setschedprio_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setschedprio_signature = rbc.typesystem.Type.fromstring("int (pthread_t, int) noexcept(true)")
pthread_setschedprio = _pthread_setschedprio_signature.toctypes()(_lib.get_pthread_setschedprio_address())
__all__.append("pthread_setschedprio")


_lib.get_pthread_getname_np_address.argtypes = ()
_lib.get_pthread_getname_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getname_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, char *, size_t) noexcept(true)")
pthread_getname_np = _pthread_getname_np_signature.toctypes()(_lib.get_pthread_getname_np_address())
__all__.append("pthread_getname_np")


_lib.get_pthread_setname_np_address.argtypes = ()
_lib.get_pthread_setname_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setname_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, const char *) noexcept(true)")
pthread_setname_np = _pthread_setname_np_signature.toctypes()(_lib.get_pthread_setname_np_address())
__all__.append("pthread_setname_np")


_lib.get_pthread_getconcurrency_address.argtypes = ()
_lib.get_pthread_getconcurrency_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getconcurrency_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
pthread_getconcurrency = _pthread_getconcurrency_signature.toctypes()(_lib.get_pthread_getconcurrency_address())
__all__.append("pthread_getconcurrency")


_lib.get_pthread_setconcurrency_address.argtypes = ()
_lib.get_pthread_setconcurrency_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setconcurrency_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
pthread_setconcurrency = _pthread_setconcurrency_signature.toctypes()(_lib.get_pthread_setconcurrency_address())
__all__.append("pthread_setconcurrency")


_lib.get_pthread_yield_address.argtypes = ()
_lib.get_pthread_yield_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_yield_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
pthread_yield = _pthread_yield_signature.toctypes()(_lib.get_pthread_yield_address())
__all__.append("pthread_yield")


_lib.get_pthread_yield_address.argtypes = ()
_lib.get_pthread_yield_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_yield_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
pthread_yield = _pthread_yield_signature.toctypes()(_lib.get_pthread_yield_address())
__all__.append("pthread_yield")


_lib.get_pthread_setaffinity_np_address.argtypes = ()
_lib.get_pthread_setaffinity_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setaffinity_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, size_t, const cpu_set_t *) noexcept(true)")
pthread_setaffinity_np = _pthread_setaffinity_np_signature.toctypes()(_lib.get_pthread_setaffinity_np_address())
__all__.append("pthread_setaffinity_np")


_lib.get_pthread_getaffinity_np_address.argtypes = ()
_lib.get_pthread_getaffinity_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getaffinity_np_signature = rbc.typesystem.Type.fromstring("int (pthread_t, size_t, cpu_set_t *) noexcept(true)")
pthread_getaffinity_np = _pthread_getaffinity_np_signature.toctypes()(_lib.get_pthread_getaffinity_np_address())
__all__.append("pthread_getaffinity_np")


_lib.get_pthread_once_address.argtypes = ()
_lib.get_pthread_once_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_once_signature = rbc.typesystem.Type.fromstring("int (pthread_once_t *, void (*)())")
pthread_once = _pthread_once_signature.toctypes()(_lib.get_pthread_once_address())
__all__.append("pthread_once")


_lib.get_pthread_setcancelstate_address.argtypes = ()
_lib.get_pthread_setcancelstate_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setcancelstate_signature = rbc.typesystem.Type.fromstring("int (int, int *)")
pthread_setcancelstate = _pthread_setcancelstate_signature.toctypes()(_lib.get_pthread_setcancelstate_address())
__all__.append("pthread_setcancelstate")


_lib.get_pthread_setcanceltype_address.argtypes = ()
_lib.get_pthread_setcanceltype_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setcanceltype_signature = rbc.typesystem.Type.fromstring("int (int, int *)")
pthread_setcanceltype = _pthread_setcanceltype_signature.toctypes()(_lib.get_pthread_setcanceltype_address())
__all__.append("pthread_setcanceltype")


_lib.get_pthread_cancel_address.argtypes = ()
_lib.get_pthread_cancel_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cancel_signature = rbc.typesystem.Type.fromstring("int (pthread_t)")
pthread_cancel = _pthread_cancel_signature.toctypes()(_lib.get_pthread_cancel_address())
__all__.append("pthread_cancel")


_lib.get_pthread_testcancel_address.argtypes = ()
_lib.get_pthread_testcancel_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_testcancel_signature = rbc.typesystem.Type.fromstring("void ()")
pthread_testcancel = _pthread_testcancel_signature.toctypes()(_lib.get_pthread_testcancel_address())
__all__.append("pthread_testcancel")


_lib.get_pthread_mutex_init_address.argtypes = ()
_lib.get_pthread_mutex_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_init_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *, const pthread_mutexattr_t *) noexcept(true)")
pthread_mutex_init = _pthread_mutex_init_signature.toctypes()(_lib.get_pthread_mutex_init_address())
__all__.append("pthread_mutex_init")


_lib.get_pthread_mutex_destroy_address.argtypes = ()
_lib.get_pthread_mutex_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_destroy = _pthread_mutex_destroy_signature.toctypes()(_lib.get_pthread_mutex_destroy_address())
__all__.append("pthread_mutex_destroy")


_lib.get_pthread_mutex_trylock_address.argtypes = ()
_lib.get_pthread_mutex_trylock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_trylock_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_trylock = _pthread_mutex_trylock_signature.toctypes()(_lib.get_pthread_mutex_trylock_address())
__all__.append("pthread_mutex_trylock")


_lib.get_pthread_mutex_lock_address.argtypes = ()
_lib.get_pthread_mutex_lock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_lock_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_lock = _pthread_mutex_lock_signature.toctypes()(_lib.get_pthread_mutex_lock_address())
__all__.append("pthread_mutex_lock")


_lib.get_pthread_mutex_timedlock_address.argtypes = ()
_lib.get_pthread_mutex_timedlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_timedlock_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *__restrict, const struct timespec *__restrict) noexcept(true)")
pthread_mutex_timedlock = _pthread_mutex_timedlock_signature.toctypes()(_lib.get_pthread_mutex_timedlock_address())
__all__.append("pthread_mutex_timedlock")


_lib.get_pthread_mutex_clocklock_address.argtypes = ()
_lib.get_pthread_mutex_clocklock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_clocklock_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true)")
pthread_mutex_clocklock = _pthread_mutex_clocklock_signature.toctypes()(_lib.get_pthread_mutex_clocklock_address())
__all__.append("pthread_mutex_clocklock")


_lib.get_pthread_mutex_unlock_address.argtypes = ()
_lib.get_pthread_mutex_unlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_unlock_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_unlock = _pthread_mutex_unlock_signature.toctypes()(_lib.get_pthread_mutex_unlock_address())
__all__.append("pthread_mutex_unlock")


_lib.get_pthread_mutex_getprioceiling_address.argtypes = ()
_lib.get_pthread_mutex_getprioceiling_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_getprioceiling_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutex_t *__restrict, int *__restrict) noexcept(true)")
pthread_mutex_getprioceiling = _pthread_mutex_getprioceiling_signature.toctypes()(_lib.get_pthread_mutex_getprioceiling_address())
__all__.append("pthread_mutex_getprioceiling")


_lib.get_pthread_mutex_setprioceiling_address.argtypes = ()
_lib.get_pthread_mutex_setprioceiling_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_setprioceiling_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *__restrict, int, int *__restrict) noexcept(true)")
pthread_mutex_setprioceiling = _pthread_mutex_setprioceiling_signature.toctypes()(_lib.get_pthread_mutex_setprioceiling_address())
__all__.append("pthread_mutex_setprioceiling")


_lib.get_pthread_mutex_consistent_address.argtypes = ()
_lib.get_pthread_mutex_consistent_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_consistent_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_consistent = _pthread_mutex_consistent_signature.toctypes()(_lib.get_pthread_mutex_consistent_address())
__all__.append("pthread_mutex_consistent")


_lib.get_pthread_mutex_consistent_np_address.argtypes = ()
_lib.get_pthread_mutex_consistent_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutex_consistent_np_signature = rbc.typesystem.Type.fromstring("int (pthread_mutex_t *) noexcept(true)")
pthread_mutex_consistent_np = _pthread_mutex_consistent_np_signature.toctypes()(_lib.get_pthread_mutex_consistent_np_address())
__all__.append("pthread_mutex_consistent_np")


_lib.get_pthread_mutexattr_init_address.argtypes = ()
_lib.get_pthread_mutexattr_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_init_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *) noexcept(true)")
pthread_mutexattr_init = _pthread_mutexattr_init_signature.toctypes()(_lib.get_pthread_mutexattr_init_address())
__all__.append("pthread_mutexattr_init")


_lib.get_pthread_mutexattr_destroy_address.argtypes = ()
_lib.get_pthread_mutexattr_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *) noexcept(true)")
pthread_mutexattr_destroy = _pthread_mutexattr_destroy_signature.toctypes()(_lib.get_pthread_mutexattr_destroy_address())
__all__.append("pthread_mutexattr_destroy")


_lib.get_pthread_mutexattr_getpshared_address.argtypes = ()
_lib.get_pthread_mutexattr_getpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_getpshared_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_mutexattr_getpshared = _pthread_mutexattr_getpshared_signature.toctypes()(_lib.get_pthread_mutexattr_getpshared_address())
__all__.append("pthread_mutexattr_getpshared")


_lib.get_pthread_mutexattr_setpshared_address.argtypes = ()
_lib.get_pthread_mutexattr_setpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_setpshared_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_setpshared = _pthread_mutexattr_setpshared_signature.toctypes()(_lib.get_pthread_mutexattr_setpshared_address())
__all__.append("pthread_mutexattr_setpshared")


_lib.get_pthread_mutexattr_gettype_address.argtypes = ()
_lib.get_pthread_mutexattr_gettype_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_gettype_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_mutexattr_gettype = _pthread_mutexattr_gettype_signature.toctypes()(_lib.get_pthread_mutexattr_gettype_address())
__all__.append("pthread_mutexattr_gettype")


_lib.get_pthread_mutexattr_settype_address.argtypes = ()
_lib.get_pthread_mutexattr_settype_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_settype_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_settype = _pthread_mutexattr_settype_signature.toctypes()(_lib.get_pthread_mutexattr_settype_address())
__all__.append("pthread_mutexattr_settype")


_lib.get_pthread_mutexattr_getprotocol_address.argtypes = ()
_lib.get_pthread_mutexattr_getprotocol_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_getprotocol_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_mutexattr_getprotocol = _pthread_mutexattr_getprotocol_signature.toctypes()(_lib.get_pthread_mutexattr_getprotocol_address())
__all__.append("pthread_mutexattr_getprotocol")


_lib.get_pthread_mutexattr_setprotocol_address.argtypes = ()
_lib.get_pthread_mutexattr_setprotocol_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_setprotocol_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_setprotocol = _pthread_mutexattr_setprotocol_signature.toctypes()(_lib.get_pthread_mutexattr_setprotocol_address())
__all__.append("pthread_mutexattr_setprotocol")


_lib.get_pthread_mutexattr_getprioceiling_address.argtypes = ()
_lib.get_pthread_mutexattr_getprioceiling_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_getprioceiling_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_mutexattr_getprioceiling = _pthread_mutexattr_getprioceiling_signature.toctypes()(_lib.get_pthread_mutexattr_getprioceiling_address())
__all__.append("pthread_mutexattr_getprioceiling")


_lib.get_pthread_mutexattr_setprioceiling_address.argtypes = ()
_lib.get_pthread_mutexattr_setprioceiling_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_setprioceiling_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_setprioceiling = _pthread_mutexattr_setprioceiling_signature.toctypes()(_lib.get_pthread_mutexattr_setprioceiling_address())
__all__.append("pthread_mutexattr_setprioceiling")


_lib.get_pthread_mutexattr_getrobust_address.argtypes = ()
_lib.get_pthread_mutexattr_getrobust_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_getrobust_signature = rbc.typesystem.Type.fromstring("int (const pthread_mutexattr_t *, int *) noexcept(true)")
pthread_mutexattr_getrobust = _pthread_mutexattr_getrobust_signature.toctypes()(_lib.get_pthread_mutexattr_getrobust_address())
__all__.append("pthread_mutexattr_getrobust")


_lib.get_pthread_mutexattr_getrobust_np_address.argtypes = ()
_lib.get_pthread_mutexattr_getrobust_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_getrobust_np_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int *) noexcept(true)")
pthread_mutexattr_getrobust_np = _pthread_mutexattr_getrobust_np_signature.toctypes()(_lib.get_pthread_mutexattr_getrobust_np_address())
__all__.append("pthread_mutexattr_getrobust_np")


_lib.get_pthread_mutexattr_setrobust_address.argtypes = ()
_lib.get_pthread_mutexattr_setrobust_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_setrobust_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_setrobust = _pthread_mutexattr_setrobust_signature.toctypes()(_lib.get_pthread_mutexattr_setrobust_address())
__all__.append("pthread_mutexattr_setrobust")


_lib.get_pthread_mutexattr_setrobust_np_address.argtypes = ()
_lib.get_pthread_mutexattr_setrobust_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_mutexattr_setrobust_np_signature = rbc.typesystem.Type.fromstring("int (pthread_mutexattr_t *, int) noexcept(true)")
pthread_mutexattr_setrobust_np = _pthread_mutexattr_setrobust_np_signature.toctypes()(_lib.get_pthread_mutexattr_setrobust_np_address())
__all__.append("pthread_mutexattr_setrobust_np")


_lib.get_pthread_rwlock_init_address.argtypes = ()
_lib.get_pthread_rwlock_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_init_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *__restrict, const pthread_rwlockattr_t *__restrict) noexcept(true)")
pthread_rwlock_init = _pthread_rwlock_init_signature.toctypes()(_lib.get_pthread_rwlock_init_address())
__all__.append("pthread_rwlock_init")


_lib.get_pthread_rwlock_destroy_address.argtypes = ()
_lib.get_pthread_rwlock_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_destroy = _pthread_rwlock_destroy_signature.toctypes()(_lib.get_pthread_rwlock_destroy_address())
__all__.append("pthread_rwlock_destroy")


_lib.get_pthread_rwlock_rdlock_address.argtypes = ()
_lib.get_pthread_rwlock_rdlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_rdlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_rdlock = _pthread_rwlock_rdlock_signature.toctypes()(_lib.get_pthread_rwlock_rdlock_address())
__all__.append("pthread_rwlock_rdlock")


_lib.get_pthread_rwlock_tryrdlock_address.argtypes = ()
_lib.get_pthread_rwlock_tryrdlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_tryrdlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_tryrdlock = _pthread_rwlock_tryrdlock_signature.toctypes()(_lib.get_pthread_rwlock_tryrdlock_address())
__all__.append("pthread_rwlock_tryrdlock")


_lib.get_pthread_rwlock_timedrdlock_address.argtypes = ()
_lib.get_pthread_rwlock_timedrdlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_timedrdlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *__restrict, const struct timespec *__restrict) noexcept(true)")
pthread_rwlock_timedrdlock = _pthread_rwlock_timedrdlock_signature.toctypes()(_lib.get_pthread_rwlock_timedrdlock_address())
__all__.append("pthread_rwlock_timedrdlock")


_lib.get_pthread_rwlock_clockrdlock_address.argtypes = ()
_lib.get_pthread_rwlock_clockrdlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_clockrdlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true)")
pthread_rwlock_clockrdlock = _pthread_rwlock_clockrdlock_signature.toctypes()(_lib.get_pthread_rwlock_clockrdlock_address())
__all__.append("pthread_rwlock_clockrdlock")


_lib.get_pthread_rwlock_wrlock_address.argtypes = ()
_lib.get_pthread_rwlock_wrlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_wrlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_wrlock = _pthread_rwlock_wrlock_signature.toctypes()(_lib.get_pthread_rwlock_wrlock_address())
__all__.append("pthread_rwlock_wrlock")


_lib.get_pthread_rwlock_trywrlock_address.argtypes = ()
_lib.get_pthread_rwlock_trywrlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_trywrlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_trywrlock = _pthread_rwlock_trywrlock_signature.toctypes()(_lib.get_pthread_rwlock_trywrlock_address())
__all__.append("pthread_rwlock_trywrlock")


_lib.get_pthread_rwlock_timedwrlock_address.argtypes = ()
_lib.get_pthread_rwlock_timedwrlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_timedwrlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *__restrict, const struct timespec *__restrict) noexcept(true)")
pthread_rwlock_timedwrlock = _pthread_rwlock_timedwrlock_signature.toctypes()(_lib.get_pthread_rwlock_timedwrlock_address())
__all__.append("pthread_rwlock_timedwrlock")


_lib.get_pthread_rwlock_clockwrlock_address.argtypes = ()
_lib.get_pthread_rwlock_clockwrlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_clockwrlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true)")
pthread_rwlock_clockwrlock = _pthread_rwlock_clockwrlock_signature.toctypes()(_lib.get_pthread_rwlock_clockwrlock_address())
__all__.append("pthread_rwlock_clockwrlock")


_lib.get_pthread_rwlock_unlock_address.argtypes = ()
_lib.get_pthread_rwlock_unlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlock_unlock_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlock_t *) noexcept(true)")
pthread_rwlock_unlock = _pthread_rwlock_unlock_signature.toctypes()(_lib.get_pthread_rwlock_unlock_address())
__all__.append("pthread_rwlock_unlock")


_lib.get_pthread_rwlockattr_init_address.argtypes = ()
_lib.get_pthread_rwlockattr_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_init_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlockattr_t *) noexcept(true)")
pthread_rwlockattr_init = _pthread_rwlockattr_init_signature.toctypes()(_lib.get_pthread_rwlockattr_init_address())
__all__.append("pthread_rwlockattr_init")


_lib.get_pthread_rwlockattr_destroy_address.argtypes = ()
_lib.get_pthread_rwlockattr_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlockattr_t *) noexcept(true)")
pthread_rwlockattr_destroy = _pthread_rwlockattr_destroy_signature.toctypes()(_lib.get_pthread_rwlockattr_destroy_address())
__all__.append("pthread_rwlockattr_destroy")


_lib.get_pthread_rwlockattr_getpshared_address.argtypes = ()
_lib.get_pthread_rwlockattr_getpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_getpshared_signature = rbc.typesystem.Type.fromstring("int (const pthread_rwlockattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_rwlockattr_getpshared = _pthread_rwlockattr_getpshared_signature.toctypes()(_lib.get_pthread_rwlockattr_getpshared_address())
__all__.append("pthread_rwlockattr_getpshared")


_lib.get_pthread_rwlockattr_setpshared_address.argtypes = ()
_lib.get_pthread_rwlockattr_setpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_setpshared_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlockattr_t *, int) noexcept(true)")
pthread_rwlockattr_setpshared = _pthread_rwlockattr_setpshared_signature.toctypes()(_lib.get_pthread_rwlockattr_setpshared_address())
__all__.append("pthread_rwlockattr_setpshared")


_lib.get_pthread_rwlockattr_getkind_np_address.argtypes = ()
_lib.get_pthread_rwlockattr_getkind_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_getkind_np_signature = rbc.typesystem.Type.fromstring("int (const pthread_rwlockattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_rwlockattr_getkind_np = _pthread_rwlockattr_getkind_np_signature.toctypes()(_lib.get_pthread_rwlockattr_getkind_np_address())
__all__.append("pthread_rwlockattr_getkind_np")


_lib.get_pthread_rwlockattr_setkind_np_address.argtypes = ()
_lib.get_pthread_rwlockattr_setkind_np_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_rwlockattr_setkind_np_signature = rbc.typesystem.Type.fromstring("int (pthread_rwlockattr_t *, int) noexcept(true)")
pthread_rwlockattr_setkind_np = _pthread_rwlockattr_setkind_np_signature.toctypes()(_lib.get_pthread_rwlockattr_setkind_np_address())
__all__.append("pthread_rwlockattr_setkind_np")


_lib.get_pthread_cond_init_address.argtypes = ()
_lib.get_pthread_cond_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_init_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *__restrict, const pthread_condattr_t *__restrict) noexcept(true)")
pthread_cond_init = _pthread_cond_init_signature.toctypes()(_lib.get_pthread_cond_init_address())
__all__.append("pthread_cond_init")


_lib.get_pthread_cond_destroy_address.argtypes = ()
_lib.get_pthread_cond_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *) noexcept(true)")
pthread_cond_destroy = _pthread_cond_destroy_signature.toctypes()(_lib.get_pthread_cond_destroy_address())
__all__.append("pthread_cond_destroy")


_lib.get_pthread_cond_signal_address.argtypes = ()
_lib.get_pthread_cond_signal_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_signal_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *) noexcept(true)")
pthread_cond_signal = _pthread_cond_signal_signature.toctypes()(_lib.get_pthread_cond_signal_address())
__all__.append("pthread_cond_signal")


_lib.get_pthread_cond_broadcast_address.argtypes = ()
_lib.get_pthread_cond_broadcast_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_broadcast_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *) noexcept(true)")
pthread_cond_broadcast = _pthread_cond_broadcast_signature.toctypes()(_lib.get_pthread_cond_broadcast_address())
__all__.append("pthread_cond_broadcast")


_lib.get_pthread_cond_wait_address.argtypes = ()
_lib.get_pthread_cond_wait_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_wait_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict)")
pthread_cond_wait = _pthread_cond_wait_signature.toctypes()(_lib.get_pthread_cond_wait_address())
__all__.append("pthread_cond_wait")


_lib.get_pthread_cond_timedwait_address.argtypes = ()
_lib.get_pthread_cond_timedwait_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_timedwait_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict, const struct timespec *__restrict)")
pthread_cond_timedwait = _pthread_cond_timedwait_signature.toctypes()(_lib.get_pthread_cond_timedwait_address())
__all__.append("pthread_cond_timedwait")


_lib.get_pthread_cond_clockwait_address.argtypes = ()
_lib.get_pthread_cond_clockwait_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_cond_clockwait_signature = rbc.typesystem.Type.fromstring("int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict, __clockid_t, const struct timespec *__restrict)")
pthread_cond_clockwait = _pthread_cond_clockwait_signature.toctypes()(_lib.get_pthread_cond_clockwait_address())
__all__.append("pthread_cond_clockwait")


_lib.get_pthread_condattr_init_address.argtypes = ()
_lib.get_pthread_condattr_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_init_signature = rbc.typesystem.Type.fromstring("int (pthread_condattr_t *) noexcept(true)")
pthread_condattr_init = _pthread_condattr_init_signature.toctypes()(_lib.get_pthread_condattr_init_address())
__all__.append("pthread_condattr_init")


_lib.get_pthread_condattr_destroy_address.argtypes = ()
_lib.get_pthread_condattr_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_condattr_t *) noexcept(true)")
pthread_condattr_destroy = _pthread_condattr_destroy_signature.toctypes()(_lib.get_pthread_condattr_destroy_address())
__all__.append("pthread_condattr_destroy")


_lib.get_pthread_condattr_getpshared_address.argtypes = ()
_lib.get_pthread_condattr_getpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_getpshared_signature = rbc.typesystem.Type.fromstring("int (const pthread_condattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_condattr_getpshared = _pthread_condattr_getpshared_signature.toctypes()(_lib.get_pthread_condattr_getpshared_address())
__all__.append("pthread_condattr_getpshared")


_lib.get_pthread_condattr_setpshared_address.argtypes = ()
_lib.get_pthread_condattr_setpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_setpshared_signature = rbc.typesystem.Type.fromstring("int (pthread_condattr_t *, int) noexcept(true)")
pthread_condattr_setpshared = _pthread_condattr_setpshared_signature.toctypes()(_lib.get_pthread_condattr_setpshared_address())
__all__.append("pthread_condattr_setpshared")


_lib.get_pthread_condattr_getclock_address.argtypes = ()
_lib.get_pthread_condattr_getclock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_getclock_signature = rbc.typesystem.Type.fromstring("int (const pthread_condattr_t *__restrict, __clockid_t *__restrict) noexcept(true)")
pthread_condattr_getclock = _pthread_condattr_getclock_signature.toctypes()(_lib.get_pthread_condattr_getclock_address())
__all__.append("pthread_condattr_getclock")


_lib.get_pthread_condattr_setclock_address.argtypes = ()
_lib.get_pthread_condattr_setclock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_condattr_setclock_signature = rbc.typesystem.Type.fromstring("int (pthread_condattr_t *, __clockid_t) noexcept(true)")
pthread_condattr_setclock = _pthread_condattr_setclock_signature.toctypes()(_lib.get_pthread_condattr_setclock_address())
__all__.append("pthread_condattr_setclock")


_lib.get_pthread_spin_init_address.argtypes = ()
_lib.get_pthread_spin_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_spin_init_signature = rbc.typesystem.Type.fromstring("int (pthread_spinlock_t *, int) noexcept(true)")
pthread_spin_init = _pthread_spin_init_signature.toctypes()(_lib.get_pthread_spin_init_address())
__all__.append("pthread_spin_init")


_lib.get_pthread_spin_destroy_address.argtypes = ()
_lib.get_pthread_spin_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_spin_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_spinlock_t *) noexcept(true)")
pthread_spin_destroy = _pthread_spin_destroy_signature.toctypes()(_lib.get_pthread_spin_destroy_address())
__all__.append("pthread_spin_destroy")


_lib.get_pthread_spin_lock_address.argtypes = ()
_lib.get_pthread_spin_lock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_spin_lock_signature = rbc.typesystem.Type.fromstring("int (pthread_spinlock_t *) noexcept(true)")
pthread_spin_lock = _pthread_spin_lock_signature.toctypes()(_lib.get_pthread_spin_lock_address())
__all__.append("pthread_spin_lock")


_lib.get_pthread_spin_trylock_address.argtypes = ()
_lib.get_pthread_spin_trylock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_spin_trylock_signature = rbc.typesystem.Type.fromstring("int (pthread_spinlock_t *) noexcept(true)")
pthread_spin_trylock = _pthread_spin_trylock_signature.toctypes()(_lib.get_pthread_spin_trylock_address())
__all__.append("pthread_spin_trylock")


_lib.get_pthread_spin_unlock_address.argtypes = ()
_lib.get_pthread_spin_unlock_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_spin_unlock_signature = rbc.typesystem.Type.fromstring("int (pthread_spinlock_t *) noexcept(true)")
pthread_spin_unlock = _pthread_spin_unlock_signature.toctypes()(_lib.get_pthread_spin_unlock_address())
__all__.append("pthread_spin_unlock")


_lib.get_pthread_barrier_init_address.argtypes = ()
_lib.get_pthread_barrier_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrier_init_signature = rbc.typesystem.Type.fromstring("int (pthread_barrier_t *__restrict, const pthread_barrierattr_t *__restrict, unsigned int) noexcept(true)")
pthread_barrier_init = _pthread_barrier_init_signature.toctypes()(_lib.get_pthread_barrier_init_address())
__all__.append("pthread_barrier_init")


_lib.get_pthread_barrier_destroy_address.argtypes = ()
_lib.get_pthread_barrier_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrier_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_barrier_t *) noexcept(true)")
pthread_barrier_destroy = _pthread_barrier_destroy_signature.toctypes()(_lib.get_pthread_barrier_destroy_address())
__all__.append("pthread_barrier_destroy")


_lib.get_pthread_barrier_wait_address.argtypes = ()
_lib.get_pthread_barrier_wait_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrier_wait_signature = rbc.typesystem.Type.fromstring("int (pthread_barrier_t *) noexcept(true)")
pthread_barrier_wait = _pthread_barrier_wait_signature.toctypes()(_lib.get_pthread_barrier_wait_address())
__all__.append("pthread_barrier_wait")


_lib.get_pthread_barrierattr_init_address.argtypes = ()
_lib.get_pthread_barrierattr_init_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrierattr_init_signature = rbc.typesystem.Type.fromstring("int (pthread_barrierattr_t *) noexcept(true)")
pthread_barrierattr_init = _pthread_barrierattr_init_signature.toctypes()(_lib.get_pthread_barrierattr_init_address())
__all__.append("pthread_barrierattr_init")


_lib.get_pthread_barrierattr_destroy_address.argtypes = ()
_lib.get_pthread_barrierattr_destroy_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrierattr_destroy_signature = rbc.typesystem.Type.fromstring("int (pthread_barrierattr_t *) noexcept(true)")
pthread_barrierattr_destroy = _pthread_barrierattr_destroy_signature.toctypes()(_lib.get_pthread_barrierattr_destroy_address())
__all__.append("pthread_barrierattr_destroy")


_lib.get_pthread_barrierattr_getpshared_address.argtypes = ()
_lib.get_pthread_barrierattr_getpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrierattr_getpshared_signature = rbc.typesystem.Type.fromstring("int (const pthread_barrierattr_t *__restrict, int *__restrict) noexcept(true)")
pthread_barrierattr_getpshared = _pthread_barrierattr_getpshared_signature.toctypes()(_lib.get_pthread_barrierattr_getpshared_address())
__all__.append("pthread_barrierattr_getpshared")


_lib.get_pthread_barrierattr_setpshared_address.argtypes = ()
_lib.get_pthread_barrierattr_setpshared_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_barrierattr_setpshared_signature = rbc.typesystem.Type.fromstring("int (pthread_barrierattr_t *, int) noexcept(true)")
pthread_barrierattr_setpshared = _pthread_barrierattr_setpshared_signature.toctypes()(_lib.get_pthread_barrierattr_setpshared_address())
__all__.append("pthread_barrierattr_setpshared")


_lib.get_pthread_key_create_address.argtypes = ()
_lib.get_pthread_key_create_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_key_create_signature = rbc.typesystem.Type.fromstring("int (pthread_key_t *, void (*)(void *)) noexcept(true)")
pthread_key_create = _pthread_key_create_signature.toctypes()(_lib.get_pthread_key_create_address())
__all__.append("pthread_key_create")


_lib.get_pthread_key_delete_address.argtypes = ()
_lib.get_pthread_key_delete_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_key_delete_signature = rbc.typesystem.Type.fromstring("int (pthread_key_t) noexcept(true)")
pthread_key_delete = _pthread_key_delete_signature.toctypes()(_lib.get_pthread_key_delete_address())
__all__.append("pthread_key_delete")


_lib.get_pthread_getspecific_address.argtypes = ()
_lib.get_pthread_getspecific_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getspecific_signature = rbc.typesystem.Type.fromstring("void *(pthread_key_t) noexcept(true)")
pthread_getspecific = _pthread_getspecific_signature.toctypes()(_lib.get_pthread_getspecific_address())
__all__.append("pthread_getspecific")


_lib.get_pthread_setspecific_address.argtypes = ()
_lib.get_pthread_setspecific_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_setspecific_signature = rbc.typesystem.Type.fromstring("int (pthread_key_t, const void *) noexcept(true)")
pthread_setspecific = _pthread_setspecific_signature.toctypes()(_lib.get_pthread_setspecific_address())
__all__.append("pthread_setspecific")


_lib.get_pthread_getcpuclockid_address.argtypes = ()
_lib.get_pthread_getcpuclockid_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_getcpuclockid_signature = rbc.typesystem.Type.fromstring("int (pthread_t, __clockid_t *) noexcept(true)")
pthread_getcpuclockid = _pthread_getcpuclockid_signature.toctypes()(_lib.get_pthread_getcpuclockid_address())
__all__.append("pthread_getcpuclockid")


_lib.get_pthread_atfork_address.argtypes = ()
_lib.get_pthread_atfork_address.restype = ctypes.c_void_p
with _target_info:
    _pthread_atfork_signature = rbc.typesystem.Type.fromstring("int (void (*)(), void (*)(), void (*)()) noexcept(true)")
pthread_atfork = _pthread_atfork_signature.toctypes()(_lib.get_pthread_atfork_address())
__all__.append("pthread_atfork")


_lib.get_atof_address.argtypes = ()
_lib.get_atof_address.restype = ctypes.c_void_p
with _target_info:
    _atof_signature = rbc.typesystem.Type.fromstring("double (const char *) noexcept(true)")
atof = _atof_signature.toctypes()(_lib.get_atof_address())
__all__.append("atof")


_lib.get_atoi_address.argtypes = ()
_lib.get_atoi_address.restype = ctypes.c_void_p
with _target_info:
    _atoi_signature = rbc.typesystem.Type.fromstring("int (const char *) noexcept(true)")
atoi = _atoi_signature.toctypes()(_lib.get_atoi_address())
__all__.append("atoi")


_lib.get_atol_address.argtypes = ()
_lib.get_atol_address.restype = ctypes.c_void_p
with _target_info:
    _atol_signature = rbc.typesystem.Type.fromstring("long (const char *) noexcept(true)")
atol = _atol_signature.toctypes()(_lib.get_atol_address())
__all__.append("atol")


_lib.get_atoll_address.argtypes = ()
_lib.get_atoll_address.restype = ctypes.c_void_p
with _target_info:
    _atoll_signature = rbc.typesystem.Type.fromstring("long long (const char *) noexcept(true)")
atoll = _atoll_signature.toctypes()(_lib.get_atoll_address())
__all__.append("atoll")


_lib.get_strtod_address.argtypes = ()
_lib.get_strtod_address.restype = ctypes.c_void_p
with _target_info:
    _strtod_signature = rbc.typesystem.Type.fromstring("double (const char *__restrict, char **__restrict) noexcept(true)")
strtod = _strtod_signature.toctypes()(_lib.get_strtod_address())
__all__.append("strtod")


_lib.get_strtof_address.argtypes = ()
_lib.get_strtof_address.restype = ctypes.c_void_p
with _target_info:
    _strtof_signature = rbc.typesystem.Type.fromstring("float (const char *__restrict, char **__restrict) noexcept(true)")
strtof = _strtof_signature.toctypes()(_lib.get_strtof_address())
__all__.append("strtof")


_lib.get_strtold_address.argtypes = ()
_lib.get_strtold_address.restype = ctypes.c_void_p
with _target_info:
    _strtold_signature = rbc.typesystem.Type.fromstring("long double (const char *__restrict, char **__restrict) noexcept(true)")
strtold = _strtold_signature.toctypes()(_lib.get_strtold_address())
__all__.append("strtold")


_lib.get_strtof32_address.argtypes = ()
_lib.get_strtof32_address.restype = ctypes.c_void_p
with _target_info:
    _strtof32_signature = rbc.typesystem.Type.fromstring("_Float32 (const char *__restrict, char **__restrict) noexcept(true)")
strtof32 = _strtof32_signature.toctypes()(_lib.get_strtof32_address())
__all__.append("strtof32")


_lib.get_strtof64_address.argtypes = ()
_lib.get_strtof64_address.restype = ctypes.c_void_p
with _target_info:
    _strtof64_signature = rbc.typesystem.Type.fromstring("_Float64 (const char *__restrict, char **__restrict) noexcept(true)")
strtof64 = _strtof64_signature.toctypes()(_lib.get_strtof64_address())
__all__.append("strtof64")


_lib.get_strtof32x_address.argtypes = ()
_lib.get_strtof32x_address.restype = ctypes.c_void_p
with _target_info:
    _strtof32x_signature = rbc.typesystem.Type.fromstring("_Float32x (const char *__restrict, char **__restrict) noexcept(true)")
strtof32x = _strtof32x_signature.toctypes()(_lib.get_strtof32x_address())
__all__.append("strtof32x")


_lib.get_strtof64x_address.argtypes = ()
_lib.get_strtof64x_address.restype = ctypes.c_void_p
with _target_info:
    _strtof64x_signature = rbc.typesystem.Type.fromstring("_Float64x (const char *__restrict, char **__restrict) noexcept(true)")
strtof64x = _strtof64x_signature.toctypes()(_lib.get_strtof64x_address())
__all__.append("strtof64x")


_lib.get_strtol_address.argtypes = ()
_lib.get_strtol_address.restype = ctypes.c_void_p
with _target_info:
    _strtol_signature = rbc.typesystem.Type.fromstring("long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtol = _strtol_signature.toctypes()(_lib.get_strtol_address())
__all__.append("strtol")


_lib.get_strtoul_address.argtypes = ()
_lib.get_strtoul_address.restype = ctypes.c_void_p
with _target_info:
    _strtoul_signature = rbc.typesystem.Type.fromstring("unsigned long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtoul = _strtoul_signature.toctypes()(_lib.get_strtoul_address())
__all__.append("strtoul")


_lib.get_strtoq_address.argtypes = ()
_lib.get_strtoq_address.restype = ctypes.c_void_p
with _target_info:
    _strtoq_signature = rbc.typesystem.Type.fromstring("long long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtoq = _strtoq_signature.toctypes()(_lib.get_strtoq_address())
__all__.append("strtoq")


_lib.get_strtouq_address.argtypes = ()
_lib.get_strtouq_address.restype = ctypes.c_void_p
with _target_info:
    _strtouq_signature = rbc.typesystem.Type.fromstring("unsigned long long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtouq = _strtouq_signature.toctypes()(_lib.get_strtouq_address())
__all__.append("strtouq")


_lib.get_strtoll_address.argtypes = ()
_lib.get_strtoll_address.restype = ctypes.c_void_p
with _target_info:
    _strtoll_signature = rbc.typesystem.Type.fromstring("long long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtoll = _strtoll_signature.toctypes()(_lib.get_strtoll_address())
__all__.append("strtoll")


_lib.get_strtoull_address.argtypes = ()
_lib.get_strtoull_address.restype = ctypes.c_void_p
with _target_info:
    _strtoull_signature = rbc.typesystem.Type.fromstring("unsigned long long (const char *__restrict, char **__restrict, int) noexcept(true)")
strtoull = _strtoull_signature.toctypes()(_lib.get_strtoull_address())
__all__.append("strtoull")


_lib.get_strfromd_address.argtypes = ()
_lib.get_strfromd_address.restype = ctypes.c_void_p
with _target_info:
    _strfromd_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, double) noexcept(true)")
strfromd = _strfromd_signature.toctypes()(_lib.get_strfromd_address())
__all__.append("strfromd")


_lib.get_strfromf_address.argtypes = ()
_lib.get_strfromf_address.restype = ctypes.c_void_p
with _target_info:
    _strfromf_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, float) noexcept(true)")
strfromf = _strfromf_signature.toctypes()(_lib.get_strfromf_address())
__all__.append("strfromf")


_lib.get_strfroml_address.argtypes = ()
_lib.get_strfroml_address.restype = ctypes.c_void_p
with _target_info:
    _strfroml_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, long double) noexcept(true)")
strfroml = _strfroml_signature.toctypes()(_lib.get_strfroml_address())
__all__.append("strfroml")


_lib.get_strfromf32_address.argtypes = ()
_lib.get_strfromf32_address.restype = ctypes.c_void_p
with _target_info:
    _strfromf32_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, _Float32) noexcept(true)")
strfromf32 = _strfromf32_signature.toctypes()(_lib.get_strfromf32_address())
__all__.append("strfromf32")


_lib.get_strfromf64_address.argtypes = ()
_lib.get_strfromf64_address.restype = ctypes.c_void_p
with _target_info:
    _strfromf64_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, _Float64) noexcept(true)")
strfromf64 = _strfromf64_signature.toctypes()(_lib.get_strfromf64_address())
__all__.append("strfromf64")


_lib.get_strfromf32x_address.argtypes = ()
_lib.get_strfromf32x_address.restype = ctypes.c_void_p
with _target_info:
    _strfromf32x_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, _Float32x) noexcept(true)")
strfromf32x = _strfromf32x_signature.toctypes()(_lib.get_strfromf32x_address())
__all__.append("strfromf32x")


_lib.get_strfromf64x_address.argtypes = ()
_lib.get_strfromf64x_address.restype = ctypes.c_void_p
with _target_info:
    _strfromf64x_signature = rbc.typesystem.Type.fromstring("int (char *, size_t, const char *, _Float64x) noexcept(true)")
strfromf64x = _strfromf64x_signature.toctypes()(_lib.get_strfromf64x_address())
__all__.append("strfromf64x")


_lib.get_strtol_l_address.argtypes = ()
_lib.get_strtol_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtol_l_signature = rbc.typesystem.Type.fromstring("long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true)")
strtol_l = _strtol_l_signature.toctypes()(_lib.get_strtol_l_address())
__all__.append("strtol_l")


_lib.get_strtoul_l_address.argtypes = ()
_lib.get_strtoul_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtoul_l_signature = rbc.typesystem.Type.fromstring("unsigned long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true)")
strtoul_l = _strtoul_l_signature.toctypes()(_lib.get_strtoul_l_address())
__all__.append("strtoul_l")


_lib.get_strtoll_l_address.argtypes = ()
_lib.get_strtoll_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtoll_l_signature = rbc.typesystem.Type.fromstring("long long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true)")
strtoll_l = _strtoll_l_signature.toctypes()(_lib.get_strtoll_l_address())
__all__.append("strtoll_l")


_lib.get_strtoull_l_address.argtypes = ()
_lib.get_strtoull_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtoull_l_signature = rbc.typesystem.Type.fromstring("unsigned long long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true)")
strtoull_l = _strtoull_l_signature.toctypes()(_lib.get_strtoull_l_address())
__all__.append("strtoull_l")


_lib.get_strtod_l_address.argtypes = ()
_lib.get_strtod_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtod_l_signature = rbc.typesystem.Type.fromstring("double (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtod_l = _strtod_l_signature.toctypes()(_lib.get_strtod_l_address())
__all__.append("strtod_l")


_lib.get_strtof_l_address.argtypes = ()
_lib.get_strtof_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtof_l_signature = rbc.typesystem.Type.fromstring("float (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtof_l = _strtof_l_signature.toctypes()(_lib.get_strtof_l_address())
__all__.append("strtof_l")


_lib.get_strtold_l_address.argtypes = ()
_lib.get_strtold_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtold_l_signature = rbc.typesystem.Type.fromstring("long double (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtold_l = _strtold_l_signature.toctypes()(_lib.get_strtold_l_address())
__all__.append("strtold_l")


_lib.get_strtof32_l_address.argtypes = ()
_lib.get_strtof32_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtof32_l_signature = rbc.typesystem.Type.fromstring("_Float32 (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtof32_l = _strtof32_l_signature.toctypes()(_lib.get_strtof32_l_address())
__all__.append("strtof32_l")


_lib.get_strtof64_l_address.argtypes = ()
_lib.get_strtof64_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtof64_l_signature = rbc.typesystem.Type.fromstring("_Float64 (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtof64_l = _strtof64_l_signature.toctypes()(_lib.get_strtof64_l_address())
__all__.append("strtof64_l")


_lib.get_strtof32x_l_address.argtypes = ()
_lib.get_strtof32x_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtof32x_l_signature = rbc.typesystem.Type.fromstring("_Float32x (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtof32x_l = _strtof32x_l_signature.toctypes()(_lib.get_strtof32x_l_address())
__all__.append("strtof32x_l")


_lib.get_strtof64x_l_address.argtypes = ()
_lib.get_strtof64x_l_address.restype = ctypes.c_void_p
with _target_info:
    _strtof64x_l_signature = rbc.typesystem.Type.fromstring("_Float64x (const char *__restrict, char **__restrict, locale_t) noexcept(true)")
strtof64x_l = _strtof64x_l_signature.toctypes()(_lib.get_strtof64x_l_address())
__all__.append("strtof64x_l")


_lib.get_l64a_address.argtypes = ()
_lib.get_l64a_address.restype = ctypes.c_void_p
with _target_info:
    _l64a_signature = rbc.typesystem.Type.fromstring("char *(long) noexcept(true)")
l64a = _l64a_signature.toctypes()(_lib.get_l64a_address())
__all__.append("l64a")


_lib.get_a64l_address.argtypes = ()
_lib.get_a64l_address.restype = ctypes.c_void_p
with _target_info:
    _a64l_signature = rbc.typesystem.Type.fromstring("long (const char *) noexcept(true)")
a64l = _a64l_signature.toctypes()(_lib.get_a64l_address())
__all__.append("a64l")


_lib.get_select_address.argtypes = ()
_lib.get_select_address.restype = ctypes.c_void_p
with _target_info:
    _select_signature = rbc.typesystem.Type.fromstring("int (int, fd_set *__restrict, fd_set *__restrict, fd_set *__restrict, struct timeval *__restrict)")
select = _select_signature.toctypes()(_lib.get_select_address())
__all__.append("select")


_lib.get_pselect_address.argtypes = ()
_lib.get_pselect_address.restype = ctypes.c_void_p
with _target_info:
    _pselect_signature = rbc.typesystem.Type.fromstring("int (int, fd_set *__restrict, fd_set *__restrict, fd_set *__restrict, const struct timespec *__restrict, const __sigset_t *__restrict)")
pselect = _pselect_signature.toctypes()(_lib.get_pselect_address())
__all__.append("pselect")


_lib.get_random_address.argtypes = ()
_lib.get_random_address.restype = ctypes.c_void_p
with _target_info:
    _random_signature = rbc.typesystem.Type.fromstring("long () noexcept(true)")
random = _random_signature.toctypes()(_lib.get_random_address())
__all__.append("random")


_lib.get_srandom_address.argtypes = ()
_lib.get_srandom_address.restype = ctypes.c_void_p
with _target_info:
    _srandom_signature = rbc.typesystem.Type.fromstring("void (unsigned int) noexcept(true)")
srandom = _srandom_signature.toctypes()(_lib.get_srandom_address())
__all__.append("srandom")


_lib.get_initstate_address.argtypes = ()
_lib.get_initstate_address.restype = ctypes.c_void_p
with _target_info:
    _initstate_signature = rbc.typesystem.Type.fromstring("char *(unsigned int, char *, size_t) noexcept(true)")
initstate = _initstate_signature.toctypes()(_lib.get_initstate_address())
__all__.append("initstate")


_lib.get_setstate_address.argtypes = ()
_lib.get_setstate_address.restype = ctypes.c_void_p
with _target_info:
    _setstate_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
setstate = _setstate_signature.toctypes()(_lib.get_setstate_address())
__all__.append("setstate")


_lib.get_random_r_address.argtypes = ()
_lib.get_random_r_address.restype = ctypes.c_void_p
with _target_info:
    _random_r_signature = rbc.typesystem.Type.fromstring("int (struct random_data *__restrict, int32_t *__restrict) noexcept(true)")
random_r = _random_r_signature.toctypes()(_lib.get_random_r_address())
__all__.append("random_r")


_lib.get_srandom_r_address.argtypes = ()
_lib.get_srandom_r_address.restype = ctypes.c_void_p
with _target_info:
    _srandom_r_signature = rbc.typesystem.Type.fromstring("int (unsigned int, struct random_data *) noexcept(true)")
srandom_r = _srandom_r_signature.toctypes()(_lib.get_srandom_r_address())
__all__.append("srandom_r")


_lib.get_initstate_r_address.argtypes = ()
_lib.get_initstate_r_address.restype = ctypes.c_void_p
with _target_info:
    _initstate_r_signature = rbc.typesystem.Type.fromstring("int (unsigned int, char *__restrict, size_t, struct random_data *__restrict) noexcept(true)")
initstate_r = _initstate_r_signature.toctypes()(_lib.get_initstate_r_address())
__all__.append("initstate_r")


_lib.get_setstate_r_address.argtypes = ()
_lib.get_setstate_r_address.restype = ctypes.c_void_p
with _target_info:
    _setstate_r_signature = rbc.typesystem.Type.fromstring("int (char *__restrict, struct random_data *__restrict) noexcept(true)")
setstate_r = _setstate_r_signature.toctypes()(_lib.get_setstate_r_address())
__all__.append("setstate_r")


_lib.get_rand_address.argtypes = ()
_lib.get_rand_address.restype = ctypes.c_void_p
with _target_info:
    _rand_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
rand = _rand_signature.toctypes()(_lib.get_rand_address())
__all__.append("rand")


_lib.get_srand_address.argtypes = ()
_lib.get_srand_address.restype = ctypes.c_void_p
with _target_info:
    _srand_signature = rbc.typesystem.Type.fromstring("void (unsigned int) noexcept(true)")
srand = _srand_signature.toctypes()(_lib.get_srand_address())
__all__.append("srand")


_lib.get_rand_r_address.argtypes = ()
_lib.get_rand_r_address.restype = ctypes.c_void_p
with _target_info:
    _rand_r_signature = rbc.typesystem.Type.fromstring("int (unsigned int *) noexcept(true)")
rand_r = _rand_r_signature.toctypes()(_lib.get_rand_r_address())
__all__.append("rand_r")


_lib.get_drand48_address.argtypes = ()
_lib.get_drand48_address.restype = ctypes.c_void_p
with _target_info:
    _drand48_signature = rbc.typesystem.Type.fromstring("double () noexcept(true)")
drand48 = _drand48_signature.toctypes()(_lib.get_drand48_address())
__all__.append("drand48")


_lib.get_erand48_address.argtypes = ()
_lib.get_erand48_address.restype = ctypes.c_void_p
with _target_info:
    _erand48_signature = rbc.typesystem.Type.fromstring("double (unsigned short *) noexcept(true)")
erand48 = _erand48_signature.toctypes()(_lib.get_erand48_address())
__all__.append("erand48")


_lib.get_lrand48_address.argtypes = ()
_lib.get_lrand48_address.restype = ctypes.c_void_p
with _target_info:
    _lrand48_signature = rbc.typesystem.Type.fromstring("long () noexcept(true)")
lrand48 = _lrand48_signature.toctypes()(_lib.get_lrand48_address())
__all__.append("lrand48")


_lib.get_nrand48_address.argtypes = ()
_lib.get_nrand48_address.restype = ctypes.c_void_p
with _target_info:
    _nrand48_signature = rbc.typesystem.Type.fromstring("long (unsigned short *) noexcept(true)")
nrand48 = _nrand48_signature.toctypes()(_lib.get_nrand48_address())
__all__.append("nrand48")


_lib.get_mrand48_address.argtypes = ()
_lib.get_mrand48_address.restype = ctypes.c_void_p
with _target_info:
    _mrand48_signature = rbc.typesystem.Type.fromstring("long () noexcept(true)")
mrand48 = _mrand48_signature.toctypes()(_lib.get_mrand48_address())
__all__.append("mrand48")


_lib.get_jrand48_address.argtypes = ()
_lib.get_jrand48_address.restype = ctypes.c_void_p
with _target_info:
    _jrand48_signature = rbc.typesystem.Type.fromstring("long (unsigned short *) noexcept(true)")
jrand48 = _jrand48_signature.toctypes()(_lib.get_jrand48_address())
__all__.append("jrand48")


_lib.get_srand48_address.argtypes = ()
_lib.get_srand48_address.restype = ctypes.c_void_p
with _target_info:
    _srand48_signature = rbc.typesystem.Type.fromstring("void (long) noexcept(true)")
srand48 = _srand48_signature.toctypes()(_lib.get_srand48_address())
__all__.append("srand48")


_lib.get_seed48_address.argtypes = ()
_lib.get_seed48_address.restype = ctypes.c_void_p
with _target_info:
    _seed48_signature = rbc.typesystem.Type.fromstring("unsigned short *(unsigned short *) noexcept(true)")
seed48 = _seed48_signature.toctypes()(_lib.get_seed48_address())
__all__.append("seed48")


_lib.get_lcong48_address.argtypes = ()
_lib.get_lcong48_address.restype = ctypes.c_void_p
with _target_info:
    _lcong48_signature = rbc.typesystem.Type.fromstring("void (unsigned short *) noexcept(true)")
lcong48 = _lcong48_signature.toctypes()(_lib.get_lcong48_address())
__all__.append("lcong48")


_lib.get_drand48_r_address.argtypes = ()
_lib.get_drand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _drand48_r_signature = rbc.typesystem.Type.fromstring("int (struct drand48_data *__restrict, double *__restrict) noexcept(true)")
drand48_r = _drand48_r_signature.toctypes()(_lib.get_drand48_r_address())
__all__.append("drand48_r")


_lib.get_erand48_r_address.argtypes = ()
_lib.get_erand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _erand48_r_signature = rbc.typesystem.Type.fromstring("int (unsigned short *, struct drand48_data *__restrict, double *__restrict) noexcept(true)")
erand48_r = _erand48_r_signature.toctypes()(_lib.get_erand48_r_address())
__all__.append("erand48_r")


_lib.get_lrand48_r_address.argtypes = ()
_lib.get_lrand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _lrand48_r_signature = rbc.typesystem.Type.fromstring("int (struct drand48_data *__restrict, long *__restrict) noexcept(true)")
lrand48_r = _lrand48_r_signature.toctypes()(_lib.get_lrand48_r_address())
__all__.append("lrand48_r")


_lib.get_nrand48_r_address.argtypes = ()
_lib.get_nrand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _nrand48_r_signature = rbc.typesystem.Type.fromstring("int (unsigned short *, struct drand48_data *__restrict, long *__restrict) noexcept(true)")
nrand48_r = _nrand48_r_signature.toctypes()(_lib.get_nrand48_r_address())
__all__.append("nrand48_r")


_lib.get_mrand48_r_address.argtypes = ()
_lib.get_mrand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _mrand48_r_signature = rbc.typesystem.Type.fromstring("int (struct drand48_data *__restrict, long *__restrict) noexcept(true)")
mrand48_r = _mrand48_r_signature.toctypes()(_lib.get_mrand48_r_address())
__all__.append("mrand48_r")


_lib.get_jrand48_r_address.argtypes = ()
_lib.get_jrand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _jrand48_r_signature = rbc.typesystem.Type.fromstring("int (unsigned short *, struct drand48_data *__restrict, long *__restrict) noexcept(true)")
jrand48_r = _jrand48_r_signature.toctypes()(_lib.get_jrand48_r_address())
__all__.append("jrand48_r")


_lib.get_srand48_r_address.argtypes = ()
_lib.get_srand48_r_address.restype = ctypes.c_void_p
with _target_info:
    _srand48_r_signature = rbc.typesystem.Type.fromstring("int (long, struct drand48_data *) noexcept(true)")
srand48_r = _srand48_r_signature.toctypes()(_lib.get_srand48_r_address())
__all__.append("srand48_r")


_lib.get_seed48_r_address.argtypes = ()
_lib.get_seed48_r_address.restype = ctypes.c_void_p
with _target_info:
    _seed48_r_signature = rbc.typesystem.Type.fromstring("int (unsigned short *, struct drand48_data *) noexcept(true)")
seed48_r = _seed48_r_signature.toctypes()(_lib.get_seed48_r_address())
__all__.append("seed48_r")


_lib.get_lcong48_r_address.argtypes = ()
_lib.get_lcong48_r_address.restype = ctypes.c_void_p
with _target_info:
    _lcong48_r_signature = rbc.typesystem.Type.fromstring("int (unsigned short *, struct drand48_data *) noexcept(true)")
lcong48_r = _lcong48_r_signature.toctypes()(_lib.get_lcong48_r_address())
__all__.append("lcong48_r")


_lib.get_malloc_address.argtypes = ()
_lib.get_malloc_address.restype = ctypes.c_void_p
with _target_info:
    _malloc_signature = rbc.typesystem.Type.fromstring("void *(size_t) noexcept(true)")
malloc = _malloc_signature.toctypes()(_lib.get_malloc_address())
__all__.append("malloc")


_lib.get_calloc_address.argtypes = ()
_lib.get_calloc_address.restype = ctypes.c_void_p
with _target_info:
    _calloc_signature = rbc.typesystem.Type.fromstring("void *(size_t, size_t) noexcept(true)")
calloc = _calloc_signature.toctypes()(_lib.get_calloc_address())
__all__.append("calloc")


_lib.get_realloc_address.argtypes = ()
_lib.get_realloc_address.restype = ctypes.c_void_p
with _target_info:
    _realloc_signature = rbc.typesystem.Type.fromstring("void *(void *, size_t) noexcept(true)")
realloc = _realloc_signature.toctypes()(_lib.get_realloc_address())
__all__.append("realloc")


_lib.get_free_address.argtypes = ()
_lib.get_free_address.restype = ctypes.c_void_p
with _target_info:
    _free_signature = rbc.typesystem.Type.fromstring("void (void *) noexcept(true)")
free = _free_signature.toctypes()(_lib.get_free_address())
__all__.append("free")


_lib.get_reallocarray_address.argtypes = ()
_lib.get_reallocarray_address.restype = ctypes.c_void_p
with _target_info:
    _reallocarray_signature = rbc.typesystem.Type.fromstring("void *(void *, size_t, size_t) noexcept(true)")
reallocarray = _reallocarray_signature.toctypes()(_lib.get_reallocarray_address())
__all__.append("reallocarray")


_lib.get_reallocarray_address.argtypes = ()
_lib.get_reallocarray_address.restype = ctypes.c_void_p
with _target_info:
    _reallocarray_signature = rbc.typesystem.Type.fromstring("void *(void *, size_t, size_t) noexcept(true)")
reallocarray = _reallocarray_signature.toctypes()(_lib.get_reallocarray_address())
__all__.append("reallocarray")


_lib.get_alloca_address.argtypes = ()
_lib.get_alloca_address.restype = ctypes.c_void_p
with _target_info:
    _alloca_signature = rbc.typesystem.Type.fromstring("void *(size_t) noexcept(true)")
alloca = _alloca_signature.toctypes()(_lib.get_alloca_address())
__all__.append("alloca")


_lib.get_valloc_address.argtypes = ()
_lib.get_valloc_address.restype = ctypes.c_void_p
with _target_info:
    _valloc_signature = rbc.typesystem.Type.fromstring("void *(size_t) noexcept(true)")
valloc = _valloc_signature.toctypes()(_lib.get_valloc_address())
__all__.append("valloc")


_lib.get_posix_memalign_address.argtypes = ()
_lib.get_posix_memalign_address.restype = ctypes.c_void_p
with _target_info:
    _posix_memalign_signature = rbc.typesystem.Type.fromstring("int (void **, size_t, size_t) noexcept(true)")
posix_memalign = _posix_memalign_signature.toctypes()(_lib.get_posix_memalign_address())
__all__.append("posix_memalign")


_lib.get_aligned_alloc_address.argtypes = ()
_lib.get_aligned_alloc_address.restype = ctypes.c_void_p
with _target_info:
    _aligned_alloc_signature = rbc.typesystem.Type.fromstring("void *(size_t, size_t) noexcept(true)")
aligned_alloc = _aligned_alloc_signature.toctypes()(_lib.get_aligned_alloc_address())
__all__.append("aligned_alloc")


_lib.get_abort_address.argtypes = ()
_lib.get_abort_address.restype = ctypes.c_void_p
with _target_info:
    _abort_signature = rbc.typesystem.Type.fromstring("void () __attribute__((noreturn)) noexcept(true)")
abort = _abort_signature.toctypes()(_lib.get_abort_address())
__all__.append("abort")


_lib.get_atexit_address.argtypes = ()
_lib.get_atexit_address.restype = ctypes.c_void_p
with _target_info:
    _atexit_signature = rbc.typesystem.Type.fromstring("int (void (*)()) noexcept(true)")
atexit = _atexit_signature.toctypes()(_lib.get_atexit_address())
__all__.append("atexit")


_lib.get_at_quick_exit_address.argtypes = ()
_lib.get_at_quick_exit_address.restype = ctypes.c_void_p
with _target_info:
    _at_quick_exit_signature = rbc.typesystem.Type.fromstring("int (void (*)()) noexcept(true)")
at_quick_exit = _at_quick_exit_signature.toctypes()(_lib.get_at_quick_exit_address())
__all__.append("at_quick_exit")


_lib.get_on_exit_address.argtypes = ()
_lib.get_on_exit_address.restype = ctypes.c_void_p
with _target_info:
    _on_exit_signature = rbc.typesystem.Type.fromstring("int (void (*)(int, void *), void *) noexcept(true)")
on_exit = _on_exit_signature.toctypes()(_lib.get_on_exit_address())
__all__.append("on_exit")


_lib.get_exit_address.argtypes = ()
_lib.get_exit_address.restype = ctypes.c_void_p
with _target_info:
    _exit_signature = rbc.typesystem.Type.fromstring("void (int) __attribute__((noreturn)) noexcept(true)")
exit = _exit_signature.toctypes()(_lib.get_exit_address())
__all__.append("exit")


_lib.get_quick_exit_address.argtypes = ()
_lib.get_quick_exit_address.restype = ctypes.c_void_p
with _target_info:
    _quick_exit_signature = rbc.typesystem.Type.fromstring("void (int) __attribute__((noreturn)) noexcept(true)")
quick_exit = _quick_exit_signature.toctypes()(_lib.get_quick_exit_address())
__all__.append("quick_exit")


_lib.get_getenv_address.argtypes = ()
_lib.get_getenv_address.restype = ctypes.c_void_p
with _target_info:
    _getenv_signature = rbc.typesystem.Type.fromstring("char *(const char *) noexcept(true)")
getenv = _getenv_signature.toctypes()(_lib.get_getenv_address())
__all__.append("getenv")


_lib.get_secure_getenv_address.argtypes = ()
_lib.get_secure_getenv_address.restype = ctypes.c_void_p
with _target_info:
    _secure_getenv_signature = rbc.typesystem.Type.fromstring("char *(const char *) noexcept(true)")
secure_getenv = _secure_getenv_signature.toctypes()(_lib.get_secure_getenv_address())
__all__.append("secure_getenv")


_lib.get_putenv_address.argtypes = ()
_lib.get_putenv_address.restype = ctypes.c_void_p
with _target_info:
    _putenv_signature = rbc.typesystem.Type.fromstring("int (char *) noexcept(true)")
putenv = _putenv_signature.toctypes()(_lib.get_putenv_address())
__all__.append("putenv")


_lib.get_setenv_address.argtypes = ()
_lib.get_setenv_address.restype = ctypes.c_void_p
with _target_info:
    _setenv_signature = rbc.typesystem.Type.fromstring("int (const char *, const char *, int) noexcept(true)")
setenv = _setenv_signature.toctypes()(_lib.get_setenv_address())
__all__.append("setenv")


_lib.get_unsetenv_address.argtypes = ()
_lib.get_unsetenv_address.restype = ctypes.c_void_p
with _target_info:
    _unsetenv_signature = rbc.typesystem.Type.fromstring("int (const char *) noexcept(true)")
unsetenv = _unsetenv_signature.toctypes()(_lib.get_unsetenv_address())
__all__.append("unsetenv")


_lib.get_clearenv_address.argtypes = ()
_lib.get_clearenv_address.restype = ctypes.c_void_p
with _target_info:
    _clearenv_signature = rbc.typesystem.Type.fromstring("int () noexcept(true)")
clearenv = _clearenv_signature.toctypes()(_lib.get_clearenv_address())
__all__.append("clearenv")


_lib.get_mktemp_address.argtypes = ()
_lib.get_mktemp_address.restype = ctypes.c_void_p
with _target_info:
    _mktemp_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
mktemp = _mktemp_signature.toctypes()(_lib.get_mktemp_address())
__all__.append("mktemp")


_lib.get_mkstemp_address.argtypes = ()
_lib.get_mkstemp_address.restype = ctypes.c_void_p
with _target_info:
    _mkstemp_signature = rbc.typesystem.Type.fromstring("int (char *)")
mkstemp = _mkstemp_signature.toctypes()(_lib.get_mkstemp_address())
__all__.append("mkstemp")


_lib.get_mkstemp64_address.argtypes = ()
_lib.get_mkstemp64_address.restype = ctypes.c_void_p
with _target_info:
    _mkstemp64_signature = rbc.typesystem.Type.fromstring("int (char *)")
mkstemp64 = _mkstemp64_signature.toctypes()(_lib.get_mkstemp64_address())
__all__.append("mkstemp64")


_lib.get_mkstemps_address.argtypes = ()
_lib.get_mkstemps_address.restype = ctypes.c_void_p
with _target_info:
    _mkstemps_signature = rbc.typesystem.Type.fromstring("int (char *, int)")
mkstemps = _mkstemps_signature.toctypes()(_lib.get_mkstemps_address())
__all__.append("mkstemps")


_lib.get_mkstemps64_address.argtypes = ()
_lib.get_mkstemps64_address.restype = ctypes.c_void_p
with _target_info:
    _mkstemps64_signature = rbc.typesystem.Type.fromstring("int (char *, int)")
mkstemps64 = _mkstemps64_signature.toctypes()(_lib.get_mkstemps64_address())
__all__.append("mkstemps64")


_lib.get_mkdtemp_address.argtypes = ()
_lib.get_mkdtemp_address.restype = ctypes.c_void_p
with _target_info:
    _mkdtemp_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
mkdtemp = _mkdtemp_signature.toctypes()(_lib.get_mkdtemp_address())
__all__.append("mkdtemp")


_lib.get_mkostemp_address.argtypes = ()
_lib.get_mkostemp_address.restype = ctypes.c_void_p
with _target_info:
    _mkostemp_signature = rbc.typesystem.Type.fromstring("int (char *, int)")
mkostemp = _mkostemp_signature.toctypes()(_lib.get_mkostemp_address())
__all__.append("mkostemp")


_lib.get_mkostemp64_address.argtypes = ()
_lib.get_mkostemp64_address.restype = ctypes.c_void_p
with _target_info:
    _mkostemp64_signature = rbc.typesystem.Type.fromstring("int (char *, int)")
mkostemp64 = _mkostemp64_signature.toctypes()(_lib.get_mkostemp64_address())
__all__.append("mkostemp64")


_lib.get_mkostemps_address.argtypes = ()
_lib.get_mkostemps_address.restype = ctypes.c_void_p
with _target_info:
    _mkostemps_signature = rbc.typesystem.Type.fromstring("int (char *, int, int)")
mkostemps = _mkostemps_signature.toctypes()(_lib.get_mkostemps_address())
__all__.append("mkostemps")


_lib.get_mkostemps64_address.argtypes = ()
_lib.get_mkostemps64_address.restype = ctypes.c_void_p
with _target_info:
    _mkostemps64_signature = rbc.typesystem.Type.fromstring("int (char *, int, int)")
mkostemps64 = _mkostemps64_signature.toctypes()(_lib.get_mkostemps64_address())
__all__.append("mkostemps64")


_lib.get_system_address.argtypes = ()
_lib.get_system_address.restype = ctypes.c_void_p
with _target_info:
    _system_signature = rbc.typesystem.Type.fromstring("int (const char *)")
system = _system_signature.toctypes()(_lib.get_system_address())
__all__.append("system")


_lib.get_canonicalize_file_name_address.argtypes = ()
_lib.get_canonicalize_file_name_address.restype = ctypes.c_void_p
with _target_info:
    _canonicalize_file_name_signature = rbc.typesystem.Type.fromstring("char *(const char *) noexcept(true)")
canonicalize_file_name = _canonicalize_file_name_signature.toctypes()(_lib.get_canonicalize_file_name_address())
__all__.append("canonicalize_file_name")


_lib.get_realpath_address.argtypes = ()
_lib.get_realpath_address.restype = ctypes.c_void_p
with _target_info:
    _realpath_signature = rbc.typesystem.Type.fromstring("char *(const char *__restrict, char *__restrict) noexcept(true)")
realpath = _realpath_signature.toctypes()(_lib.get_realpath_address())
__all__.append("realpath")


_lib.get_bsearch_address.argtypes = ()
_lib.get_bsearch_address.restype = ctypes.c_void_p
with _target_info:
    _bsearch_signature = rbc.typesystem.Type.fromstring("void *(const void *, const void *, size_t, size_t, __compar_fn_t)")
bsearch = _bsearch_signature.toctypes()(_lib.get_bsearch_address())
__all__.append("bsearch")


_lib.get_qsort_address.argtypes = ()
_lib.get_qsort_address.restype = ctypes.c_void_p
with _target_info:
    _qsort_signature = rbc.typesystem.Type.fromstring("void (void *, size_t, size_t, __compar_fn_t)")
qsort = _qsort_signature.toctypes()(_lib.get_qsort_address())
__all__.append("qsort")


_lib.get_qsort_r_address.argtypes = ()
_lib.get_qsort_r_address.restype = ctypes.c_void_p
with _target_info:
    _qsort_r_signature = rbc.typesystem.Type.fromstring("void (void *, size_t, size_t, __compar_d_fn_t, void *)")
qsort_r = _qsort_r_signature.toctypes()(_lib.get_qsort_r_address())
__all__.append("qsort_r")


_lib.get_abs_address.argtypes = ()
_lib.get_abs_address.restype = ctypes.c_void_p
with _target_info:
    _abs_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
abs = _abs_signature.toctypes()(_lib.get_abs_address())
__all__.append("abs")


_lib.get_labs_address.argtypes = ()
_lib.get_labs_address.restype = ctypes.c_void_p
with _target_info:
    _labs_signature = rbc.typesystem.Type.fromstring("long (long) noexcept(true)")
labs = _labs_signature.toctypes()(_lib.get_labs_address())
__all__.append("labs")


_lib.get_llabs_address.argtypes = ()
_lib.get_llabs_address.restype = ctypes.c_void_p
with _target_info:
    _llabs_signature = rbc.typesystem.Type.fromstring("long long (long long) noexcept(true)")
llabs = _llabs_signature.toctypes()(_lib.get_llabs_address())
__all__.append("llabs")


_lib.get_div_address.argtypes = ()
_lib.get_div_address.restype = ctypes.c_void_p
with _target_info:
    _div_signature = rbc.typesystem.Type.fromstring("div_t (int, int) noexcept(true)")
div = _div_signature.toctypes()(_lib.get_div_address())
__all__.append("div")


_lib.get_ldiv_address.argtypes = ()
_lib.get_ldiv_address.restype = ctypes.c_void_p
with _target_info:
    _ldiv_signature = rbc.typesystem.Type.fromstring("ldiv_t (long, long) noexcept(true)")
ldiv = _ldiv_signature.toctypes()(_lib.get_ldiv_address())
__all__.append("ldiv")


_lib.get_lldiv_address.argtypes = ()
_lib.get_lldiv_address.restype = ctypes.c_void_p
with _target_info:
    _lldiv_signature = rbc.typesystem.Type.fromstring("lldiv_t (long long, long long) noexcept(true)")
lldiv = _lldiv_signature.toctypes()(_lib.get_lldiv_address())
__all__.append("lldiv")


_lib.get_ecvt_address.argtypes = ()
_lib.get_ecvt_address.restype = ctypes.c_void_p
with _target_info:
    _ecvt_signature = rbc.typesystem.Type.fromstring("char *(double, int, int *__restrict, int *__restrict) noexcept(true)")
ecvt = _ecvt_signature.toctypes()(_lib.get_ecvt_address())
__all__.append("ecvt")


_lib.get_fcvt_address.argtypes = ()
_lib.get_fcvt_address.restype = ctypes.c_void_p
with _target_info:
    _fcvt_signature = rbc.typesystem.Type.fromstring("char *(double, int, int *__restrict, int *__restrict) noexcept(true)")
fcvt = _fcvt_signature.toctypes()(_lib.get_fcvt_address())
__all__.append("fcvt")


_lib.get_gcvt_address.argtypes = ()
_lib.get_gcvt_address.restype = ctypes.c_void_p
with _target_info:
    _gcvt_signature = rbc.typesystem.Type.fromstring("char *(double, int, char *) noexcept(true)")
gcvt = _gcvt_signature.toctypes()(_lib.get_gcvt_address())
__all__.append("gcvt")


_lib.get_qecvt_address.argtypes = ()
_lib.get_qecvt_address.restype = ctypes.c_void_p
with _target_info:
    _qecvt_signature = rbc.typesystem.Type.fromstring("char *(long double, int, int *__restrict, int *__restrict) noexcept(true)")
qecvt = _qecvt_signature.toctypes()(_lib.get_qecvt_address())
__all__.append("qecvt")


_lib.get_qfcvt_address.argtypes = ()
_lib.get_qfcvt_address.restype = ctypes.c_void_p
with _target_info:
    _qfcvt_signature = rbc.typesystem.Type.fromstring("char *(long double, int, int *__restrict, int *__restrict) noexcept(true)")
qfcvt = _qfcvt_signature.toctypes()(_lib.get_qfcvt_address())
__all__.append("qfcvt")


_lib.get_qgcvt_address.argtypes = ()
_lib.get_qgcvt_address.restype = ctypes.c_void_p
with _target_info:
    _qgcvt_signature = rbc.typesystem.Type.fromstring("char *(long double, int, char *) noexcept(true)")
qgcvt = _qgcvt_signature.toctypes()(_lib.get_qgcvt_address())
__all__.append("qgcvt")


_lib.get_ecvt_r_address.argtypes = ()
_lib.get_ecvt_r_address.restype = ctypes.c_void_p
with _target_info:
    _ecvt_r_signature = rbc.typesystem.Type.fromstring("int (double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true)")
ecvt_r = _ecvt_r_signature.toctypes()(_lib.get_ecvt_r_address())
__all__.append("ecvt_r")


_lib.get_fcvt_r_address.argtypes = ()
_lib.get_fcvt_r_address.restype = ctypes.c_void_p
with _target_info:
    _fcvt_r_signature = rbc.typesystem.Type.fromstring("int (double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true)")
fcvt_r = _fcvt_r_signature.toctypes()(_lib.get_fcvt_r_address())
__all__.append("fcvt_r")


_lib.get_qecvt_r_address.argtypes = ()
_lib.get_qecvt_r_address.restype = ctypes.c_void_p
with _target_info:
    _qecvt_r_signature = rbc.typesystem.Type.fromstring("int (long double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true)")
qecvt_r = _qecvt_r_signature.toctypes()(_lib.get_qecvt_r_address())
__all__.append("qecvt_r")


_lib.get_qfcvt_r_address.argtypes = ()
_lib.get_qfcvt_r_address.restype = ctypes.c_void_p
with _target_info:
    _qfcvt_r_signature = rbc.typesystem.Type.fromstring("int (long double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true)")
qfcvt_r = _qfcvt_r_signature.toctypes()(_lib.get_qfcvt_r_address())
__all__.append("qfcvt_r")


_lib.get_mblen_address.argtypes = ()
_lib.get_mblen_address.restype = ctypes.c_void_p
with _target_info:
    _mblen_signature = rbc.typesystem.Type.fromstring("int (const char *, size_t) noexcept(true)")
mblen = _mblen_signature.toctypes()(_lib.get_mblen_address())
__all__.append("mblen")


_lib.get_mbtowc_address.argtypes = ()
_lib.get_mbtowc_address.restype = ctypes.c_void_p
with _target_info:
    _mbtowc_signature = rbc.typesystem.Type.fromstring("int (wchar_t *__restrict, const char *__restrict, size_t) noexcept(true)")
mbtowc = _mbtowc_signature.toctypes()(_lib.get_mbtowc_address())
__all__.append("mbtowc")


_lib.get_wctomb_address.argtypes = ()
_lib.get_wctomb_address.restype = ctypes.c_void_p
with _target_info:
    _wctomb_signature = rbc.typesystem.Type.fromstring("int (char *, wchar_t) noexcept(true)")
wctomb = _wctomb_signature.toctypes()(_lib.get_wctomb_address())
__all__.append("wctomb")


_lib.get_mbstowcs_address.argtypes = ()
_lib.get_mbstowcs_address.restype = ctypes.c_void_p
with _target_info:
    _mbstowcs_signature = rbc.typesystem.Type.fromstring("size_t (wchar_t *__restrict, const char *__restrict, size_t) noexcept(true)")
mbstowcs = _mbstowcs_signature.toctypes()(_lib.get_mbstowcs_address())
__all__.append("mbstowcs")


_lib.get_wcstombs_address.argtypes = ()
_lib.get_wcstombs_address.restype = ctypes.c_void_p
with _target_info:
    _wcstombs_signature = rbc.typesystem.Type.fromstring("size_t (char *__restrict, const wchar_t *__restrict, size_t) noexcept(true)")
wcstombs = _wcstombs_signature.toctypes()(_lib.get_wcstombs_address())
__all__.append("wcstombs")


_lib.get_rpmatch_address.argtypes = ()
_lib.get_rpmatch_address.restype = ctypes.c_void_p
with _target_info:
    _rpmatch_signature = rbc.typesystem.Type.fromstring("int (const char *) noexcept(true)")
rpmatch = _rpmatch_signature.toctypes()(_lib.get_rpmatch_address())
__all__.append("rpmatch")


_lib.get_getsubopt_address.argtypes = ()
_lib.get_getsubopt_address.restype = ctypes.c_void_p
with _target_info:
    _getsubopt_signature = rbc.typesystem.Type.fromstring("int (char **__restrict, char *const *__restrict, char **__restrict) noexcept(true)")
getsubopt = _getsubopt_signature.toctypes()(_lib.get_getsubopt_address())
__all__.append("getsubopt")


_lib.get_posix_openpt_address.argtypes = ()
_lib.get_posix_openpt_address.restype = ctypes.c_void_p
with _target_info:
    _posix_openpt_signature = rbc.typesystem.Type.fromstring("int (int)")
posix_openpt = _posix_openpt_signature.toctypes()(_lib.get_posix_openpt_address())
__all__.append("posix_openpt")


_lib.get_grantpt_address.argtypes = ()
_lib.get_grantpt_address.restype = ctypes.c_void_p
with _target_info:
    _grantpt_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
grantpt = _grantpt_signature.toctypes()(_lib.get_grantpt_address())
__all__.append("grantpt")


_lib.get_unlockpt_address.argtypes = ()
_lib.get_unlockpt_address.restype = ctypes.c_void_p
with _target_info:
    _unlockpt_signature = rbc.typesystem.Type.fromstring("int (int) noexcept(true)")
unlockpt = _unlockpt_signature.toctypes()(_lib.get_unlockpt_address())
__all__.append("unlockpt")


_lib.get_ptsname_address.argtypes = ()
_lib.get_ptsname_address.restype = ctypes.c_void_p
with _target_info:
    _ptsname_signature = rbc.typesystem.Type.fromstring("char *(int) noexcept(true)")
ptsname = _ptsname_signature.toctypes()(_lib.get_ptsname_address())
__all__.append("ptsname")


_lib.get_ptsname_r_address.argtypes = ()
_lib.get_ptsname_r_address.restype = ctypes.c_void_p
with _target_info:
    _ptsname_r_signature = rbc.typesystem.Type.fromstring("int (int, char *, size_t) noexcept(true)")
ptsname_r = _ptsname_r_signature.toctypes()(_lib.get_ptsname_r_address())
__all__.append("ptsname_r")


_lib.get_getpt_address.argtypes = ()
_lib.get_getpt_address.restype = ctypes.c_void_p
with _target_info:
    _getpt_signature = rbc.typesystem.Type.fromstring("int ()")
getpt = _getpt_signature.toctypes()(_lib.get_getpt_address())
__all__.append("getpt")


_lib.get_getloadavg_address.argtypes = ()
_lib.get_getloadavg_address.restype = ctypes.c_void_p
with _target_info:
    _getloadavg_signature = rbc.typesystem.Type.fromstring("int (double *, int) noexcept(true)")
getloadavg = _getloadavg_signature.toctypes()(_lib.get_getloadavg_address())
__all__.append("getloadavg")


_lib.get_remove_address.argtypes = ()
_lib.get_remove_address.restype = ctypes.c_void_p
with _target_info:
    _remove_signature = rbc.typesystem.Type.fromstring("int (const char *) noexcept(true)")
remove = _remove_signature.toctypes()(_lib.get_remove_address())
__all__.append("remove")


_lib.get_rename_address.argtypes = ()
_lib.get_rename_address.restype = ctypes.c_void_p
with _target_info:
    _rename_signature = rbc.typesystem.Type.fromstring("int (const char *, const char *) noexcept(true)")
rename = _rename_signature.toctypes()(_lib.get_rename_address())
__all__.append("rename")


_lib.get_renameat_address.argtypes = ()
_lib.get_renameat_address.restype = ctypes.c_void_p
with _target_info:
    _renameat_signature = rbc.typesystem.Type.fromstring("int (int, const char *, int, const char *) noexcept(true)")
renameat = _renameat_signature.toctypes()(_lib.get_renameat_address())
__all__.append("renameat")


_lib.get_renameat2_address.argtypes = ()
_lib.get_renameat2_address.restype = ctypes.c_void_p
with _target_info:
    _renameat2_signature = rbc.typesystem.Type.fromstring("int (int, const char *, int, const char *, unsigned int) noexcept(true)")
renameat2 = _renameat2_signature.toctypes()(_lib.get_renameat2_address())
__all__.append("renameat2")


_lib.get_fclose_address.argtypes = ()
_lib.get_fclose_address.restype = ctypes.c_void_p
with _target_info:
    _fclose_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
fclose = _fclose_signature.toctypes()(_lib.get_fclose_address())
__all__.append("fclose")


_lib.get_tmpfile_address.argtypes = ()
_lib.get_tmpfile_address.restype = ctypes.c_void_p
with _target_info:
    _tmpfile_signature = rbc.typesystem.Type.fromstring("FILE *()")
tmpfile = _tmpfile_signature.toctypes()(_lib.get_tmpfile_address())
__all__.append("tmpfile")


_lib.get_tmpfile64_address.argtypes = ()
_lib.get_tmpfile64_address.restype = ctypes.c_void_p
with _target_info:
    _tmpfile64_signature = rbc.typesystem.Type.fromstring("FILE *()")
tmpfile64 = _tmpfile64_signature.toctypes()(_lib.get_tmpfile64_address())
__all__.append("tmpfile64")


_lib.get_tmpnam_address.argtypes = ()
_lib.get_tmpnam_address.restype = ctypes.c_void_p
with _target_info:
    _tmpnam_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
tmpnam = _tmpnam_signature.toctypes()(_lib.get_tmpnam_address())
__all__.append("tmpnam")


_lib.get_tmpnam_r_address.argtypes = ()
_lib.get_tmpnam_r_address.restype = ctypes.c_void_p
with _target_info:
    _tmpnam_r_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
tmpnam_r = _tmpnam_r_signature.toctypes()(_lib.get_tmpnam_r_address())
__all__.append("tmpnam_r")


_lib.get_tempnam_address.argtypes = ()
_lib.get_tempnam_address.restype = ctypes.c_void_p
with _target_info:
    _tempnam_signature = rbc.typesystem.Type.fromstring("char *(const char *, const char *) noexcept(true)")
tempnam = _tempnam_signature.toctypes()(_lib.get_tempnam_address())
__all__.append("tempnam")


_lib.get_fflush_address.argtypes = ()
_lib.get_fflush_address.restype = ctypes.c_void_p
with _target_info:
    _fflush_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
fflush = _fflush_signature.toctypes()(_lib.get_fflush_address())
__all__.append("fflush")


_lib.get_fflush_unlocked_address.argtypes = ()
_lib.get_fflush_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fflush_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
fflush_unlocked = _fflush_unlocked_signature.toctypes()(_lib.get_fflush_unlocked_address())
__all__.append("fflush_unlocked")


_lib.get_fcloseall_address.argtypes = ()
_lib.get_fcloseall_address.restype = ctypes.c_void_p
with _target_info:
    _fcloseall_signature = rbc.typesystem.Type.fromstring("int ()")
fcloseall = _fcloseall_signature.toctypes()(_lib.get_fcloseall_address())
__all__.append("fcloseall")


_lib.get_fopen_address.argtypes = ()
_lib.get_fopen_address.restype = ctypes.c_void_p
with _target_info:
    _fopen_signature = rbc.typesystem.Type.fromstring("FILE *(const char *__restrict, const char *__restrict)")
fopen = _fopen_signature.toctypes()(_lib.get_fopen_address())
__all__.append("fopen")


_lib.get_freopen_address.argtypes = ()
_lib.get_freopen_address.restype = ctypes.c_void_p
with _target_info:
    _freopen_signature = rbc.typesystem.Type.fromstring("FILE *(const char *__restrict, const char *__restrict, FILE *__restrict)")
freopen = _freopen_signature.toctypes()(_lib.get_freopen_address())
__all__.append("freopen")


_lib.get_fopen64_address.argtypes = ()
_lib.get_fopen64_address.restype = ctypes.c_void_p
with _target_info:
    _fopen64_signature = rbc.typesystem.Type.fromstring("FILE *(const char *__restrict, const char *__restrict)")
fopen64 = _fopen64_signature.toctypes()(_lib.get_fopen64_address())
__all__.append("fopen64")


_lib.get_freopen64_address.argtypes = ()
_lib.get_freopen64_address.restype = ctypes.c_void_p
with _target_info:
    _freopen64_signature = rbc.typesystem.Type.fromstring("FILE *(const char *__restrict, const char *__restrict, FILE *__restrict)")
freopen64 = _freopen64_signature.toctypes()(_lib.get_freopen64_address())
__all__.append("freopen64")


_lib.get_fdopen_address.argtypes = ()
_lib.get_fdopen_address.restype = ctypes.c_void_p
with _target_info:
    _fdopen_signature = rbc.typesystem.Type.fromstring("FILE *(int, const char *) noexcept(true)")
fdopen = _fdopen_signature.toctypes()(_lib.get_fdopen_address())
__all__.append("fdopen")


_lib.get_fopencookie_address.argtypes = ()
_lib.get_fopencookie_address.restype = ctypes.c_void_p
with _target_info:
    _fopencookie_signature = rbc.typesystem.Type.fromstring("FILE *(void *__restrict, const char *__restrict, cookie_io_functions_t) noexcept(true)")
fopencookie = _fopencookie_signature.toctypes()(_lib.get_fopencookie_address())
__all__.append("fopencookie")


_lib.get_fmemopen_address.argtypes = ()
_lib.get_fmemopen_address.restype = ctypes.c_void_p
with _target_info:
    _fmemopen_signature = rbc.typesystem.Type.fromstring("FILE *(void *, size_t, const char *) noexcept(true)")
fmemopen = _fmemopen_signature.toctypes()(_lib.get_fmemopen_address())
__all__.append("fmemopen")


_lib.get_open_memstream_address.argtypes = ()
_lib.get_open_memstream_address.restype = ctypes.c_void_p
with _target_info:
    _open_memstream_signature = rbc.typesystem.Type.fromstring("FILE *(char **, size_t *) noexcept(true)")
open_memstream = _open_memstream_signature.toctypes()(_lib.get_open_memstream_address())
__all__.append("open_memstream")


_lib.get_open_wmemstream_address.argtypes = ()
_lib.get_open_wmemstream_address.restype = ctypes.c_void_p
with _target_info:
    _open_wmemstream_signature = rbc.typesystem.Type.fromstring("__FILE *(wchar_t **, size_t *) noexcept(true)")
open_wmemstream = _open_wmemstream_signature.toctypes()(_lib.get_open_wmemstream_address())
__all__.append("open_wmemstream")


_lib.get_setbuf_address.argtypes = ()
_lib.get_setbuf_address.restype = ctypes.c_void_p
with _target_info:
    _setbuf_signature = rbc.typesystem.Type.fromstring("void (FILE *__restrict, char *__restrict) noexcept(true)")
setbuf = _setbuf_signature.toctypes()(_lib.get_setbuf_address())
__all__.append("setbuf")


_lib.get_setvbuf_address.argtypes = ()
_lib.get_setvbuf_address.restype = ctypes.c_void_p
with _target_info:
    _setvbuf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, char *__restrict, int, size_t) noexcept(true)")
setvbuf = _setvbuf_signature.toctypes()(_lib.get_setvbuf_address())
__all__.append("setvbuf")


_lib.get_setbuffer_address.argtypes = ()
_lib.get_setbuffer_address.restype = ctypes.c_void_p
with _target_info:
    _setbuffer_signature = rbc.typesystem.Type.fromstring("void (FILE *__restrict, char *__restrict, size_t) noexcept(true)")
setbuffer = _setbuffer_signature.toctypes()(_lib.get_setbuffer_address())
__all__.append("setbuffer")


_lib.get_setlinebuf_address.argtypes = ()
_lib.get_setlinebuf_address.restype = ctypes.c_void_p
with _target_info:
    _setlinebuf_signature = rbc.typesystem.Type.fromstring("void (FILE *) noexcept(true)")
setlinebuf = _setlinebuf_signature.toctypes()(_lib.get_setlinebuf_address())
__all__.append("setlinebuf")


_lib.get_fprintf_address.argtypes = ()
_lib.get_fprintf_address.restype = ctypes.c_void_p
with _target_info:
    _fprintf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, ...)")
fprintf = _fprintf_signature.toctypes()(_lib.get_fprintf_address())
__all__.append("fprintf")


_lib.get_printf_address.argtypes = ()
_lib.get_printf_address.restype = ctypes.c_void_p
with _target_info:
    _printf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, ...)")
printf = _printf_signature.toctypes()(_lib.get_printf_address())
__all__.append("printf")


_lib.get_sprintf_address.argtypes = ()
_lib.get_sprintf_address.restype = ctypes.c_void_p
with _target_info:
    _sprintf_signature = rbc.typesystem.Type.fromstring("int (char *__restrict, const char *__restrict, ...) noexcept(true)")
sprintf = _sprintf_signature.toctypes()(_lib.get_sprintf_address())
__all__.append("sprintf")


_lib.get_vfprintf_address.argtypes = ()
_lib.get_vfprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vfprintf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, __va_list_tag *)")
vfprintf = _vfprintf_signature.toctypes()(_lib.get_vfprintf_address())
__all__.append("vfprintf")


_lib.get_vprintf_address.argtypes = ()
_lib.get_vprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vprintf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, __va_list_tag *)")
vprintf = _vprintf_signature.toctypes()(_lib.get_vprintf_address())
__all__.append("vprintf")


_lib.get_vsprintf_address.argtypes = ()
_lib.get_vsprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vsprintf_signature = rbc.typesystem.Type.fromstring("int (char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true)")
vsprintf = _vsprintf_signature.toctypes()(_lib.get_vsprintf_address())
__all__.append("vsprintf")


_lib.get_snprintf_address.argtypes = ()
_lib.get_snprintf_address.restype = ctypes.c_void_p
with _target_info:
    _snprintf_signature = rbc.typesystem.Type.fromstring("int (char *__restrict, size_t, const char *__restrict, ...) noexcept(true)")
snprintf = _snprintf_signature.toctypes()(_lib.get_snprintf_address())
__all__.append("snprintf")


_lib.get_vsnprintf_address.argtypes = ()
_lib.get_vsnprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vsnprintf_signature = rbc.typesystem.Type.fromstring("int (char *__restrict, size_t, const char *__restrict, __va_list_tag *) noexcept(true)")
vsnprintf = _vsnprintf_signature.toctypes()(_lib.get_vsnprintf_address())
__all__.append("vsnprintf")


_lib.get_vasprintf_address.argtypes = ()
_lib.get_vasprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vasprintf_signature = rbc.typesystem.Type.fromstring("int (char **__restrict, const char *__restrict, __va_list_tag *) noexcept(true)")
vasprintf = _vasprintf_signature.toctypes()(_lib.get_vasprintf_address())
__all__.append("vasprintf")


_lib.get_asprintf_address.argtypes = ()
_lib.get_asprintf_address.restype = ctypes.c_void_p
with _target_info:
    _asprintf_signature = rbc.typesystem.Type.fromstring("int (char **__restrict, const char *__restrict, ...) noexcept(true)")
asprintf = _asprintf_signature.toctypes()(_lib.get_asprintf_address())
__all__.append("asprintf")


_lib.get_vdprintf_address.argtypes = ()
_lib.get_vdprintf_address.restype = ctypes.c_void_p
with _target_info:
    _vdprintf_signature = rbc.typesystem.Type.fromstring("int (int, const char *__restrict, __va_list_tag *)")
vdprintf = _vdprintf_signature.toctypes()(_lib.get_vdprintf_address())
__all__.append("vdprintf")


_lib.get_dprintf_address.argtypes = ()
_lib.get_dprintf_address.restype = ctypes.c_void_p
with _target_info:
    _dprintf_signature = rbc.typesystem.Type.fromstring("int (int, const char *__restrict, ...)")
dprintf = _dprintf_signature.toctypes()(_lib.get_dprintf_address())
__all__.append("dprintf")


_lib.get_fscanf_address.argtypes = ()
_lib.get_fscanf_address.restype = ctypes.c_void_p
with _target_info:
    _fscanf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, ...)")
fscanf = _fscanf_signature.toctypes()(_lib.get_fscanf_address())
__all__.append("fscanf")


_lib.get_scanf_address.argtypes = ()
_lib.get_scanf_address.restype = ctypes.c_void_p
with _target_info:
    _scanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, ...)")
scanf = _scanf_signature.toctypes()(_lib.get_scanf_address())
__all__.append("scanf")


_lib.get_sscanf_address.argtypes = ()
_lib.get_sscanf_address.restype = ctypes.c_void_p
with _target_info:
    _sscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, const char *__restrict, ...) noexcept(true)")
sscanf = _sscanf_signature.toctypes()(_lib.get_sscanf_address())
__all__.append("sscanf")


_lib.get_fscanf_address.argtypes = ()
_lib.get_fscanf_address.restype = ctypes.c_void_p
with _target_info:
    _fscanf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, ...)")
fscanf = _fscanf_signature.toctypes()(_lib.get_fscanf_address())
__all__.append("fscanf")


_lib.get_scanf_address.argtypes = ()
_lib.get_scanf_address.restype = ctypes.c_void_p
with _target_info:
    _scanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, ...)")
scanf = _scanf_signature.toctypes()(_lib.get_scanf_address())
__all__.append("scanf")


_lib.get_sscanf_address.argtypes = ()
_lib.get_sscanf_address.restype = ctypes.c_void_p
with _target_info:
    _sscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, const char *__restrict, ...) noexcept(true)")
sscanf = _sscanf_signature.toctypes()(_lib.get_sscanf_address())
__all__.append("sscanf")


_lib.get_vfscanf_address.argtypes = ()
_lib.get_vfscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vfscanf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, __va_list_tag *)")
vfscanf = _vfscanf_signature.toctypes()(_lib.get_vfscanf_address())
__all__.append("vfscanf")


_lib.get_vscanf_address.argtypes = ()
_lib.get_vscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, __va_list_tag *)")
vscanf = _vscanf_signature.toctypes()(_lib.get_vscanf_address())
__all__.append("vscanf")


_lib.get_vsscanf_address.argtypes = ()
_lib.get_vsscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vsscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true)")
vsscanf = _vsscanf_signature.toctypes()(_lib.get_vsscanf_address())
__all__.append("vsscanf")


_lib.get_vfscanf_address.argtypes = ()
_lib.get_vfscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vfscanf_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, const char *__restrict, __va_list_tag *)")
vfscanf = _vfscanf_signature.toctypes()(_lib.get_vfscanf_address())
__all__.append("vfscanf")


_lib.get_vscanf_address.argtypes = ()
_lib.get_vscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, __va_list_tag *)")
vscanf = _vscanf_signature.toctypes()(_lib.get_vscanf_address())
__all__.append("vscanf")


_lib.get_vsscanf_address.argtypes = ()
_lib.get_vsscanf_address.restype = ctypes.c_void_p
with _target_info:
    _vsscanf_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true)")
vsscanf = _vsscanf_signature.toctypes()(_lib.get_vsscanf_address())
__all__.append("vsscanf")


_lib.get_fgetc_address.argtypes = ()
_lib.get_fgetc_address.restype = ctypes.c_void_p
with _target_info:
    _fgetc_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
fgetc = _fgetc_signature.toctypes()(_lib.get_fgetc_address())
__all__.append("fgetc")


_lib.get_getc_address.argtypes = ()
_lib.get_getc_address.restype = ctypes.c_void_p
with _target_info:
    _getc_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
getc = _getc_signature.toctypes()(_lib.get_getc_address())
__all__.append("getc")


_lib.get_getchar_address.argtypes = ()
_lib.get_getchar_address.restype = ctypes.c_void_p
with _target_info:
    _getchar_signature = rbc.typesystem.Type.fromstring("int ()")
getchar = _getchar_signature.toctypes()(_lib.get_getchar_address())
__all__.append("getchar")


_lib.get_getc_unlocked_address.argtypes = ()
_lib.get_getc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _getc_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
getc_unlocked = _getc_unlocked_signature.toctypes()(_lib.get_getc_unlocked_address())
__all__.append("getc_unlocked")


_lib.get_getchar_unlocked_address.argtypes = ()
_lib.get_getchar_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _getchar_unlocked_signature = rbc.typesystem.Type.fromstring("int ()")
getchar_unlocked = _getchar_unlocked_signature.toctypes()(_lib.get_getchar_unlocked_address())
__all__.append("getchar_unlocked")


_lib.get_fgetc_unlocked_address.argtypes = ()
_lib.get_fgetc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fgetc_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
fgetc_unlocked = _fgetc_unlocked_signature.toctypes()(_lib.get_fgetc_unlocked_address())
__all__.append("fgetc_unlocked")


_lib.get_fputc_address.argtypes = ()
_lib.get_fputc_address.restype = ctypes.c_void_p
with _target_info:
    _fputc_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
fputc = _fputc_signature.toctypes()(_lib.get_fputc_address())
__all__.append("fputc")


_lib.get_putc_address.argtypes = ()
_lib.get_putc_address.restype = ctypes.c_void_p
with _target_info:
    _putc_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
putc = _putc_signature.toctypes()(_lib.get_putc_address())
__all__.append("putc")


_lib.get_putchar_address.argtypes = ()
_lib.get_putchar_address.restype = ctypes.c_void_p
with _target_info:
    _putchar_signature = rbc.typesystem.Type.fromstring("int (int)")
putchar = _putchar_signature.toctypes()(_lib.get_putchar_address())
__all__.append("putchar")


_lib.get_fputc_unlocked_address.argtypes = ()
_lib.get_fputc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fputc_unlocked_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
fputc_unlocked = _fputc_unlocked_signature.toctypes()(_lib.get_fputc_unlocked_address())
__all__.append("fputc_unlocked")


_lib.get_putc_unlocked_address.argtypes = ()
_lib.get_putc_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _putc_unlocked_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
putc_unlocked = _putc_unlocked_signature.toctypes()(_lib.get_putc_unlocked_address())
__all__.append("putc_unlocked")


_lib.get_putchar_unlocked_address.argtypes = ()
_lib.get_putchar_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _putchar_unlocked_signature = rbc.typesystem.Type.fromstring("int (int)")
putchar_unlocked = _putchar_unlocked_signature.toctypes()(_lib.get_putchar_unlocked_address())
__all__.append("putchar_unlocked")


_lib.get_getw_address.argtypes = ()
_lib.get_getw_address.restype = ctypes.c_void_p
with _target_info:
    _getw_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
getw = _getw_signature.toctypes()(_lib.get_getw_address())
__all__.append("getw")


_lib.get_putw_address.argtypes = ()
_lib.get_putw_address.restype = ctypes.c_void_p
with _target_info:
    _putw_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
putw = _putw_signature.toctypes()(_lib.get_putw_address())
__all__.append("putw")


_lib.get_fgets_address.argtypes = ()
_lib.get_fgets_address.restype = ctypes.c_void_p
with _target_info:
    _fgets_signature = rbc.typesystem.Type.fromstring("char *(char *__restrict, int, FILE *__restrict)")
fgets = _fgets_signature.toctypes()(_lib.get_fgets_address())
__all__.append("fgets")


_lib.get_fgets_unlocked_address.argtypes = ()
_lib.get_fgets_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fgets_unlocked_signature = rbc.typesystem.Type.fromstring("char *(char *__restrict, int, FILE *__restrict)")
fgets_unlocked = _fgets_unlocked_signature.toctypes()(_lib.get_fgets_unlocked_address())
__all__.append("fgets_unlocked")


_lib.get_getdelim_address.argtypes = ()
_lib.get_getdelim_address.restype = ctypes.c_void_p
with _target_info:
    _getdelim_signature = rbc.typesystem.Type.fromstring("__ssize_t (char **__restrict, size_t *__restrict, int, FILE *__restrict)")
getdelim = _getdelim_signature.toctypes()(_lib.get_getdelim_address())
__all__.append("getdelim")


_lib.get_getline_address.argtypes = ()
_lib.get_getline_address.restype = ctypes.c_void_p
with _target_info:
    _getline_signature = rbc.typesystem.Type.fromstring("__ssize_t (char **__restrict, size_t *__restrict, FILE *__restrict)")
getline = _getline_signature.toctypes()(_lib.get_getline_address())
__all__.append("getline")


_lib.get_fputs_address.argtypes = ()
_lib.get_fputs_address.restype = ctypes.c_void_p
with _target_info:
    _fputs_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, FILE *__restrict)")
fputs = _fputs_signature.toctypes()(_lib.get_fputs_address())
__all__.append("fputs")


_lib.get_puts_address.argtypes = ()
_lib.get_puts_address.restype = ctypes.c_void_p
with _target_info:
    _puts_signature = rbc.typesystem.Type.fromstring("int (const char *)")
puts = _puts_signature.toctypes()(_lib.get_puts_address())
__all__.append("puts")


_lib.get_ungetc_address.argtypes = ()
_lib.get_ungetc_address.restype = ctypes.c_void_p
with _target_info:
    _ungetc_signature = rbc.typesystem.Type.fromstring("int (int, FILE *)")
ungetc = _ungetc_signature.toctypes()(_lib.get_ungetc_address())
__all__.append("ungetc")


_lib.get_fread_address.argtypes = ()
_lib.get_fread_address.restype = ctypes.c_void_p
with _target_info:
    _fread_signature = rbc.typesystem.Type.fromstring("size_t (void *__restrict, size_t, size_t, FILE *__restrict)")
fread = _fread_signature.toctypes()(_lib.get_fread_address())
__all__.append("fread")


_lib.get_fwrite_address.argtypes = ()
_lib.get_fwrite_address.restype = ctypes.c_void_p
with _target_info:
    _fwrite_signature = rbc.typesystem.Type.fromstring("size_t (const void *__restrict, size_t, size_t, FILE *__restrict)")
fwrite = _fwrite_signature.toctypes()(_lib.get_fwrite_address())
__all__.append("fwrite")


_lib.get_fputs_unlocked_address.argtypes = ()
_lib.get_fputs_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fputs_unlocked_signature = rbc.typesystem.Type.fromstring("int (const char *__restrict, FILE *__restrict)")
fputs_unlocked = _fputs_unlocked_signature.toctypes()(_lib.get_fputs_unlocked_address())
__all__.append("fputs_unlocked")


_lib.get_fread_unlocked_address.argtypes = ()
_lib.get_fread_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fread_unlocked_signature = rbc.typesystem.Type.fromstring("size_t (void *__restrict, size_t, size_t, FILE *__restrict)")
fread_unlocked = _fread_unlocked_signature.toctypes()(_lib.get_fread_unlocked_address())
__all__.append("fread_unlocked")


_lib.get_fwrite_unlocked_address.argtypes = ()
_lib.get_fwrite_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fwrite_unlocked_signature = rbc.typesystem.Type.fromstring("size_t (const void *__restrict, size_t, size_t, FILE *__restrict)")
fwrite_unlocked = _fwrite_unlocked_signature.toctypes()(_lib.get_fwrite_unlocked_address())
__all__.append("fwrite_unlocked")


_lib.get_fseek_address.argtypes = ()
_lib.get_fseek_address.restype = ctypes.c_void_p
with _target_info:
    _fseek_signature = rbc.typesystem.Type.fromstring("int (FILE *, long, int)")
fseek = _fseek_signature.toctypes()(_lib.get_fseek_address())
__all__.append("fseek")


_lib.get_ftell_address.argtypes = ()
_lib.get_ftell_address.restype = ctypes.c_void_p
with _target_info:
    _ftell_signature = rbc.typesystem.Type.fromstring("long (FILE *)")
ftell = _ftell_signature.toctypes()(_lib.get_ftell_address())
__all__.append("ftell")


_lib.get_rewind_address.argtypes = ()
_lib.get_rewind_address.restype = ctypes.c_void_p
with _target_info:
    _rewind_signature = rbc.typesystem.Type.fromstring("void (FILE *)")
rewind = _rewind_signature.toctypes()(_lib.get_rewind_address())
__all__.append("rewind")


_lib.get_fseeko_address.argtypes = ()
_lib.get_fseeko_address.restype = ctypes.c_void_p
with _target_info:
    _fseeko_signature = rbc.typesystem.Type.fromstring("int (FILE *, __off_t, int)")
fseeko = _fseeko_signature.toctypes()(_lib.get_fseeko_address())
__all__.append("fseeko")


_lib.get_ftello_address.argtypes = ()
_lib.get_ftello_address.restype = ctypes.c_void_p
with _target_info:
    _ftello_signature = rbc.typesystem.Type.fromstring("__off_t (FILE *)")
ftello = _ftello_signature.toctypes()(_lib.get_ftello_address())
__all__.append("ftello")


_lib.get_fgetpos_address.argtypes = ()
_lib.get_fgetpos_address.restype = ctypes.c_void_p
with _target_info:
    _fgetpos_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, fpos_t *__restrict)")
fgetpos = _fgetpos_signature.toctypes()(_lib.get_fgetpos_address())
__all__.append("fgetpos")


_lib.get_fsetpos_address.argtypes = ()
_lib.get_fsetpos_address.restype = ctypes.c_void_p
with _target_info:
    _fsetpos_signature = rbc.typesystem.Type.fromstring("int (FILE *, const fpos_t *)")
fsetpos = _fsetpos_signature.toctypes()(_lib.get_fsetpos_address())
__all__.append("fsetpos")


_lib.get_fseeko64_address.argtypes = ()
_lib.get_fseeko64_address.restype = ctypes.c_void_p
with _target_info:
    _fseeko64_signature = rbc.typesystem.Type.fromstring("int (FILE *, __off64_t, int)")
fseeko64 = _fseeko64_signature.toctypes()(_lib.get_fseeko64_address())
__all__.append("fseeko64")


_lib.get_ftello64_address.argtypes = ()
_lib.get_ftello64_address.restype = ctypes.c_void_p
with _target_info:
    _ftello64_signature = rbc.typesystem.Type.fromstring("__off64_t (FILE *)")
ftello64 = _ftello64_signature.toctypes()(_lib.get_ftello64_address())
__all__.append("ftello64")


_lib.get_fgetpos64_address.argtypes = ()
_lib.get_fgetpos64_address.restype = ctypes.c_void_p
with _target_info:
    _fgetpos64_signature = rbc.typesystem.Type.fromstring("int (FILE *__restrict, fpos64_t *__restrict)")
fgetpos64 = _fgetpos64_signature.toctypes()(_lib.get_fgetpos64_address())
__all__.append("fgetpos64")


_lib.get_fsetpos64_address.argtypes = ()
_lib.get_fsetpos64_address.restype = ctypes.c_void_p
with _target_info:
    _fsetpos64_signature = rbc.typesystem.Type.fromstring("int (FILE *, const fpos64_t *)")
fsetpos64 = _fsetpos64_signature.toctypes()(_lib.get_fsetpos64_address())
__all__.append("fsetpos64")


_lib.get_clearerr_address.argtypes = ()
_lib.get_clearerr_address.restype = ctypes.c_void_p
with _target_info:
    _clearerr_signature = rbc.typesystem.Type.fromstring("void (FILE *) noexcept(true)")
clearerr = _clearerr_signature.toctypes()(_lib.get_clearerr_address())
__all__.append("clearerr")


_lib.get_feof_address.argtypes = ()
_lib.get_feof_address.restype = ctypes.c_void_p
with _target_info:
    _feof_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
feof = _feof_signature.toctypes()(_lib.get_feof_address())
__all__.append("feof")


_lib.get_ferror_address.argtypes = ()
_lib.get_ferror_address.restype = ctypes.c_void_p
with _target_info:
    _ferror_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
ferror = _ferror_signature.toctypes()(_lib.get_ferror_address())
__all__.append("ferror")


_lib.get_clearerr_unlocked_address.argtypes = ()
_lib.get_clearerr_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _clearerr_unlocked_signature = rbc.typesystem.Type.fromstring("void (FILE *) noexcept(true)")
clearerr_unlocked = _clearerr_unlocked_signature.toctypes()(_lib.get_clearerr_unlocked_address())
__all__.append("clearerr_unlocked")


_lib.get_feof_unlocked_address.argtypes = ()
_lib.get_feof_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _feof_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
feof_unlocked = _feof_unlocked_signature.toctypes()(_lib.get_feof_unlocked_address())
__all__.append("feof_unlocked")


_lib.get_ferror_unlocked_address.argtypes = ()
_lib.get_ferror_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _ferror_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
ferror_unlocked = _ferror_unlocked_signature.toctypes()(_lib.get_ferror_unlocked_address())
__all__.append("ferror_unlocked")


_lib.get_perror_address.argtypes = ()
_lib.get_perror_address.restype = ctypes.c_void_p
with _target_info:
    _perror_signature = rbc.typesystem.Type.fromstring("void (const char *)")
perror = _perror_signature.toctypes()(_lib.get_perror_address())
__all__.append("perror")


_lib.get_fileno_address.argtypes = ()
_lib.get_fileno_address.restype = ctypes.c_void_p
with _target_info:
    _fileno_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
fileno = _fileno_signature.toctypes()(_lib.get_fileno_address())
__all__.append("fileno")


_lib.get_fileno_unlocked_address.argtypes = ()
_lib.get_fileno_unlocked_address.restype = ctypes.c_void_p
with _target_info:
    _fileno_unlocked_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
fileno_unlocked = _fileno_unlocked_signature.toctypes()(_lib.get_fileno_unlocked_address())
__all__.append("fileno_unlocked")


_lib.get_pclose_address.argtypes = ()
_lib.get_pclose_address.restype = ctypes.c_void_p
with _target_info:
    _pclose_signature = rbc.typesystem.Type.fromstring("int (FILE *)")
pclose = _pclose_signature.toctypes()(_lib.get_pclose_address())
__all__.append("pclose")


_lib.get_popen_address.argtypes = ()
_lib.get_popen_address.restype = ctypes.c_void_p
with _target_info:
    _popen_signature = rbc.typesystem.Type.fromstring("FILE *(const char *, const char *)")
popen = _popen_signature.toctypes()(_lib.get_popen_address())
__all__.append("popen")


_lib.get_ctermid_address.argtypes = ()
_lib.get_ctermid_address.restype = ctypes.c_void_p
with _target_info:
    _ctermid_signature = rbc.typesystem.Type.fromstring("char *(char *) noexcept(true)")
ctermid = _ctermid_signature.toctypes()(_lib.get_ctermid_address())
__all__.append("ctermid")


_lib.get_cuserid_address.argtypes = ()
_lib.get_cuserid_address.restype = ctypes.c_void_p
with _target_info:
    _cuserid_signature = rbc.typesystem.Type.fromstring("char *(char *)")
cuserid = _cuserid_signature.toctypes()(_lib.get_cuserid_address())
__all__.append("cuserid")


_lib.get_obstack_printf_address.argtypes = ()
_lib.get_obstack_printf_address.restype = ctypes.c_void_p
with _target_info:
    _obstack_printf_signature = rbc.typesystem.Type.fromstring("int (struct obstack *__restrict, const char *__restrict, ...) noexcept(true)")
obstack_printf = _obstack_printf_signature.toctypes()(_lib.get_obstack_printf_address())
__all__.append("obstack_printf")


_lib.get_obstack_vprintf_address.argtypes = ()
_lib.get_obstack_vprintf_address.restype = ctypes.c_void_p
with _target_info:
    _obstack_vprintf_signature = rbc.typesystem.Type.fromstring("int (struct obstack *__restrict, const char *__restrict, __va_list_tag *) noexcept(true)")
obstack_vprintf = _obstack_vprintf_signature.toctypes()(_lib.get_obstack_vprintf_address())
__all__.append("obstack_vprintf")


_lib.get_flockfile_address.argtypes = ()
_lib.get_flockfile_address.restype = ctypes.c_void_p
with _target_info:
    _flockfile_signature = rbc.typesystem.Type.fromstring("void (FILE *) noexcept(true)")
flockfile = _flockfile_signature.toctypes()(_lib.get_flockfile_address())
__all__.append("flockfile")


_lib.get_ftrylockfile_address.argtypes = ()
_lib.get_ftrylockfile_address.restype = ctypes.c_void_p
with _target_info:
    _ftrylockfile_signature = rbc.typesystem.Type.fromstring("int (FILE *) noexcept(true)")
ftrylockfile = _ftrylockfile_signature.toctypes()(_lib.get_ftrylockfile_address())
__all__.append("ftrylockfile")


_lib.get_funlockfile_address.argtypes = ()
_lib.get_funlockfile_address.restype = ctypes.c_void_p
with _target_info:
    _funlockfile_signature = rbc.typesystem.Type.fromstring("void (FILE *) noexcept(true)")
funlockfile = _funlockfile_signature.toctypes()(_lib.get_funlockfile_address())
__all__.append("funlockfile")


_lib.get_iswalnum_address.argtypes = ()
_lib.get_iswalnum_address.restype = ctypes.c_void_p
with _target_info:
    _iswalnum_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswalnum = _iswalnum_signature.toctypes()(_lib.get_iswalnum_address())
__all__.append("iswalnum")


_lib.get_iswalpha_address.argtypes = ()
_lib.get_iswalpha_address.restype = ctypes.c_void_p
with _target_info:
    _iswalpha_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswalpha = _iswalpha_signature.toctypes()(_lib.get_iswalpha_address())
__all__.append("iswalpha")


_lib.get_iswcntrl_address.argtypes = ()
_lib.get_iswcntrl_address.restype = ctypes.c_void_p
with _target_info:
    _iswcntrl_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswcntrl = _iswcntrl_signature.toctypes()(_lib.get_iswcntrl_address())
__all__.append("iswcntrl")


_lib.get_iswdigit_address.argtypes = ()
_lib.get_iswdigit_address.restype = ctypes.c_void_p
with _target_info:
    _iswdigit_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswdigit = _iswdigit_signature.toctypes()(_lib.get_iswdigit_address())
__all__.append("iswdigit")


_lib.get_iswgraph_address.argtypes = ()
_lib.get_iswgraph_address.restype = ctypes.c_void_p
with _target_info:
    _iswgraph_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswgraph = _iswgraph_signature.toctypes()(_lib.get_iswgraph_address())
__all__.append("iswgraph")


_lib.get_iswlower_address.argtypes = ()
_lib.get_iswlower_address.restype = ctypes.c_void_p
with _target_info:
    _iswlower_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswlower = _iswlower_signature.toctypes()(_lib.get_iswlower_address())
__all__.append("iswlower")


_lib.get_iswprint_address.argtypes = ()
_lib.get_iswprint_address.restype = ctypes.c_void_p
with _target_info:
    _iswprint_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswprint = _iswprint_signature.toctypes()(_lib.get_iswprint_address())
__all__.append("iswprint")


_lib.get_iswpunct_address.argtypes = ()
_lib.get_iswpunct_address.restype = ctypes.c_void_p
with _target_info:
    _iswpunct_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswpunct = _iswpunct_signature.toctypes()(_lib.get_iswpunct_address())
__all__.append("iswpunct")


_lib.get_iswspace_address.argtypes = ()
_lib.get_iswspace_address.restype = ctypes.c_void_p
with _target_info:
    _iswspace_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswspace = _iswspace_signature.toctypes()(_lib.get_iswspace_address())
__all__.append("iswspace")


_lib.get_iswupper_address.argtypes = ()
_lib.get_iswupper_address.restype = ctypes.c_void_p
with _target_info:
    _iswupper_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswupper = _iswupper_signature.toctypes()(_lib.get_iswupper_address())
__all__.append("iswupper")


_lib.get_iswxdigit_address.argtypes = ()
_lib.get_iswxdigit_address.restype = ctypes.c_void_p
with _target_info:
    _iswxdigit_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswxdigit = _iswxdigit_signature.toctypes()(_lib.get_iswxdigit_address())
__all__.append("iswxdigit")


_lib.get_iswblank_address.argtypes = ()
_lib.get_iswblank_address.restype = ctypes.c_void_p
with _target_info:
    _iswblank_signature = rbc.typesystem.Type.fromstring("int (wint_t) noexcept(true)")
iswblank = _iswblank_signature.toctypes()(_lib.get_iswblank_address())
__all__.append("iswblank")


_lib.get_wctype_address.argtypes = ()
_lib.get_wctype_address.restype = ctypes.c_void_p
with _target_info:
    _wctype_signature = rbc.typesystem.Type.fromstring("wctype_t (const char *) noexcept(true)")
wctype = _wctype_signature.toctypes()(_lib.get_wctype_address())
__all__.append("wctype")


_lib.get_iswctype_address.argtypes = ()
_lib.get_iswctype_address.restype = ctypes.c_void_p
with _target_info:
    _iswctype_signature = rbc.typesystem.Type.fromstring("int (wint_t, wctype_t) noexcept(true)")
iswctype = _iswctype_signature.toctypes()(_lib.get_iswctype_address())
__all__.append("iswctype")


_lib.get_towlower_address.argtypes = ()
_lib.get_towlower_address.restype = ctypes.c_void_p
with _target_info:
    _towlower_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t) noexcept(true)")
towlower = _towlower_signature.toctypes()(_lib.get_towlower_address())
__all__.append("towlower")


_lib.get_towupper_address.argtypes = ()
_lib.get_towupper_address.restype = ctypes.c_void_p
with _target_info:
    _towupper_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t) noexcept(true)")
towupper = _towupper_signature.toctypes()(_lib.get_towupper_address())
__all__.append("towupper")


_lib.get_wctrans_address.argtypes = ()
_lib.get_wctrans_address.restype = ctypes.c_void_p
with _target_info:
    _wctrans_signature = rbc.typesystem.Type.fromstring("wctrans_t (const char *) noexcept(true)")
wctrans = _wctrans_signature.toctypes()(_lib.get_wctrans_address())
__all__.append("wctrans")


_lib.get_towctrans_address.argtypes = ()
_lib.get_towctrans_address.restype = ctypes.c_void_p
with _target_info:
    _towctrans_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t, wctrans_t) noexcept(true)")
towctrans = _towctrans_signature.toctypes()(_lib.get_towctrans_address())
__all__.append("towctrans")


_lib.get_iswalnum_l_address.argtypes = ()
_lib.get_iswalnum_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswalnum_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswalnum_l = _iswalnum_l_signature.toctypes()(_lib.get_iswalnum_l_address())
__all__.append("iswalnum_l")


_lib.get_iswalpha_l_address.argtypes = ()
_lib.get_iswalpha_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswalpha_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswalpha_l = _iswalpha_l_signature.toctypes()(_lib.get_iswalpha_l_address())
__all__.append("iswalpha_l")


_lib.get_iswcntrl_l_address.argtypes = ()
_lib.get_iswcntrl_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswcntrl_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswcntrl_l = _iswcntrl_l_signature.toctypes()(_lib.get_iswcntrl_l_address())
__all__.append("iswcntrl_l")


_lib.get_iswdigit_l_address.argtypes = ()
_lib.get_iswdigit_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswdigit_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswdigit_l = _iswdigit_l_signature.toctypes()(_lib.get_iswdigit_l_address())
__all__.append("iswdigit_l")


_lib.get_iswgraph_l_address.argtypes = ()
_lib.get_iswgraph_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswgraph_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswgraph_l = _iswgraph_l_signature.toctypes()(_lib.get_iswgraph_l_address())
__all__.append("iswgraph_l")


_lib.get_iswlower_l_address.argtypes = ()
_lib.get_iswlower_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswlower_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswlower_l = _iswlower_l_signature.toctypes()(_lib.get_iswlower_l_address())
__all__.append("iswlower_l")


_lib.get_iswprint_l_address.argtypes = ()
_lib.get_iswprint_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswprint_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswprint_l = _iswprint_l_signature.toctypes()(_lib.get_iswprint_l_address())
__all__.append("iswprint_l")


_lib.get_iswpunct_l_address.argtypes = ()
_lib.get_iswpunct_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswpunct_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswpunct_l = _iswpunct_l_signature.toctypes()(_lib.get_iswpunct_l_address())
__all__.append("iswpunct_l")


_lib.get_iswspace_l_address.argtypes = ()
_lib.get_iswspace_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswspace_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswspace_l = _iswspace_l_signature.toctypes()(_lib.get_iswspace_l_address())
__all__.append("iswspace_l")


_lib.get_iswupper_l_address.argtypes = ()
_lib.get_iswupper_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswupper_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswupper_l = _iswupper_l_signature.toctypes()(_lib.get_iswupper_l_address())
__all__.append("iswupper_l")


_lib.get_iswxdigit_l_address.argtypes = ()
_lib.get_iswxdigit_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswxdigit_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswxdigit_l = _iswxdigit_l_signature.toctypes()(_lib.get_iswxdigit_l_address())
__all__.append("iswxdigit_l")


_lib.get_iswblank_l_address.argtypes = ()
_lib.get_iswblank_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswblank_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, locale_t) noexcept(true)")
iswblank_l = _iswblank_l_signature.toctypes()(_lib.get_iswblank_l_address())
__all__.append("iswblank_l")


_lib.get_wctype_l_address.argtypes = ()
_lib.get_wctype_l_address.restype = ctypes.c_void_p
with _target_info:
    _wctype_l_signature = rbc.typesystem.Type.fromstring("wctype_t (const char *, locale_t) noexcept(true)")
wctype_l = _wctype_l_signature.toctypes()(_lib.get_wctype_l_address())
__all__.append("wctype_l")


_lib.get_iswctype_l_address.argtypes = ()
_lib.get_iswctype_l_address.restype = ctypes.c_void_p
with _target_info:
    _iswctype_l_signature = rbc.typesystem.Type.fromstring("int (wint_t, wctype_t, locale_t) noexcept(true)")
iswctype_l = _iswctype_l_signature.toctypes()(_lib.get_iswctype_l_address())
__all__.append("iswctype_l")


_lib.get_towlower_l_address.argtypes = ()
_lib.get_towlower_l_address.restype = ctypes.c_void_p
with _target_info:
    _towlower_l_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t, locale_t) noexcept(true)")
towlower_l = _towlower_l_signature.toctypes()(_lib.get_towlower_l_address())
__all__.append("towlower_l")


_lib.get_towupper_l_address.argtypes = ()
_lib.get_towupper_l_address.restype = ctypes.c_void_p
with _target_info:
    _towupper_l_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t, locale_t) noexcept(true)")
towupper_l = _towupper_l_signature.toctypes()(_lib.get_towupper_l_address())
__all__.append("towupper_l")


_lib.get_wctrans_l_address.argtypes = ()
_lib.get_wctrans_l_address.restype = ctypes.c_void_p
with _target_info:
    _wctrans_l_signature = rbc.typesystem.Type.fromstring("wctrans_t (const char *, locale_t) noexcept(true)")
wctrans_l = _wctrans_l_signature.toctypes()(_lib.get_wctrans_l_address())
__all__.append("wctrans_l")


_lib.get_towctrans_l_address.argtypes = ()
_lib.get_towctrans_l_address.restype = ctypes.c_void_p
with _target_info:
    _towctrans_l_signature = rbc.typesystem.Type.fromstring("wint_t (wint_t, wctrans_t, locale_t) noexcept(true)")
towctrans_l = _towctrans_l_signature.toctypes()(_lib.get_towctrans_l_address())
__all__.append("towctrans_l")
