#include <memory>
#include <cstdint>
#include "pot1d.hpp"

extern "C" intptr_t get_wcscpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscpy));
}


extern "C" intptr_t get_wcsncpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsncpy));
}


extern "C" intptr_t get_wcscat_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscat));
}


extern "C" intptr_t get_wcsncat_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsncat));
}


extern "C" intptr_t get_wcscmp_address() {
  /* int (const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscmp));
}


extern "C" intptr_t get_wcsncmp_address() {
  /* int (const wchar_t *, const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsncmp));
}


extern "C" intptr_t get_wcscasecmp_address() {
  /* int (const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscasecmp));
}


extern "C" intptr_t get_wcsncasecmp_address() {
  /* int (const wchar_t *, const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsncasecmp));
}


extern "C" intptr_t get_wcscasecmp_l_address() {
  /* int (const wchar_t *, const wchar_t *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscasecmp_l));
}


extern "C" intptr_t get_wcsncasecmp_l_address() {
  /* int (const wchar_t *, const wchar_t *, size_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsncasecmp_l));
}


extern "C" intptr_t get_wcscoll_address() {
  /* int (const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscoll));
}


extern "C" intptr_t get_wcsxfrm_address() {
  /* size_t (wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsxfrm));
}


extern "C" intptr_t get_wcscoll_l_address() {
  /* int (const wchar_t *, const wchar_t *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscoll_l));
}


extern "C" intptr_t get_wcsxfrm_l_address() {
  /* size_t (wchar_t *, const wchar_t *, size_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsxfrm_l));
}


extern "C" intptr_t get_wcsdup_address() {
  /* wchar_t *(const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsdup));
}


extern "C" intptr_t get_wcschr_address() {
  /* wchar_t *(const wchar_t *, wchar_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcschr));
}


extern "C" intptr_t get_wcsrchr_address() {
  /* wchar_t *(const wchar_t *, wchar_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsrchr));
}


extern "C" intptr_t get_wcschrnul_address() {
  /* wchar_t *(const wchar_t *, wchar_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcschrnul));
}


extern "C" intptr_t get_wcscspn_address() {
  /* size_t (const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcscspn));
}


extern "C" intptr_t get_wcsspn_address() {
  /* size_t (const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsspn));
}


extern "C" intptr_t get_wcspbrk_address() {
  /* wchar_t *(const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcspbrk));
}


extern "C" intptr_t get_wcsstr_address() {
  /* wchar_t *(const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsstr));
}


extern "C" intptr_t get_wcstok_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstok));
}


extern "C" intptr_t get_wcslen_address() {
  /* size_t (const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcslen));
}


extern "C" intptr_t get_wcswcs_address() {
  /* wchar_t *(const wchar_t *, const wchar_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcswcs));
}


extern "C" intptr_t get_wcsnlen_address() {
  /* size_t (const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsnlen));
}


extern "C" intptr_t get_wmemchr_address() {
  /* wchar_t *(const wchar_t *, wchar_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmemchr));
}


extern "C" intptr_t get_wmemcmp_address() {
  /* int (const wchar_t *, const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmemcmp));
}


extern "C" intptr_t get_wmemcpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmemcpy));
}


extern "C" intptr_t get_wmemmove_address() {
  /* wchar_t *(wchar_t *, const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmemmove));
}


extern "C" intptr_t get_wmemset_address() {
  /* wchar_t *(wchar_t *, wchar_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmemset));
}


extern "C" intptr_t get_wmempcpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wmempcpy));
}


extern "C" intptr_t get_btowc_address() {
  /* wint_t (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(btowc));
}


extern "C" intptr_t get_wctob_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctob));
}


extern "C" intptr_t get_mbsinit_address() {
  /* int (const mbstate_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbsinit));
}


extern "C" intptr_t get_mbrtowc_address() {
  /* size_t (wchar_t *__restrict, const char *__restrict, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbrtowc));
}


extern "C" intptr_t get_wcrtomb_address() {
  /* size_t (char *__restrict, wchar_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcrtomb));
}


extern "C" intptr_t get_mbrlen_address() {
  /* size_t (const char *__restrict, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbrlen));
}


extern "C" intptr_t get_mbsrtowcs_address() {
  /* size_t (wchar_t *__restrict, const char **__restrict, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbsrtowcs));
}


extern "C" intptr_t get_wcsrtombs_address() {
  /* size_t (char *__restrict, const wchar_t **__restrict, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsrtombs));
}


extern "C" intptr_t get_mbsnrtowcs_address() {
  /* size_t (wchar_t *__restrict, const char **__restrict, size_t, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbsnrtowcs));
}


extern "C" intptr_t get_wcsnrtombs_address() {
  /* size_t (char *__restrict, const wchar_t **__restrict, size_t, size_t, mbstate_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsnrtombs));
}


extern "C" intptr_t get_wcwidth_address() {
  /* int (wchar_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcwidth));
}


extern "C" intptr_t get_wcswidth_address() {
  /* int (const wchar_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcswidth));
}


extern "C" intptr_t get_wcstod_address() {
  /* double (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstod));
}


extern "C" intptr_t get_wcstof_address() {
  /* float (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof));
}


extern "C" intptr_t get_wcstold_address() {
  /* long double (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstold));
}


extern "C" intptr_t get_wcstof32_address() {
  /* _Float32 (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof32));
}


extern "C" intptr_t get_wcstof64_address() {
  /* _Float64 (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof64));
}


extern "C" intptr_t get_wcstof32x_address() {
  /* _Float32x (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof32x));
}


extern "C" intptr_t get_wcstof64x_address() {
  /* _Float64x (const wchar_t *__restrict, wchar_t **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof64x));
}


extern "C" intptr_t get_wcstol_address() {
  /* long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstol));
}


extern "C" intptr_t get_wcstoul_address() {
  /* unsigned long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoul));
}


extern "C" intptr_t get_wcstoll_address() {
  /* long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoll));
}


extern "C" intptr_t get_wcstoull_address() {
  /* unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoull));
}


extern "C" intptr_t get_wcstoq_address() {
  /* long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoq));
}


extern "C" intptr_t get_wcstouq_address() {
  /* unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstouq));
}


extern "C" intptr_t get_wcstol_l_address() {
  /* long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstol_l));
}


extern "C" intptr_t get_wcstoul_l_address() {
  /* unsigned long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoul_l));
}


extern "C" intptr_t get_wcstoll_l_address() {
  /* long long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoll_l));
}


extern "C" intptr_t get_wcstoull_l_address() {
  /* unsigned long long (const wchar_t *__restrict, wchar_t **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstoull_l));
}


extern "C" intptr_t get_wcstod_l_address() {
  /* double (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstod_l));
}


extern "C" intptr_t get_wcstof_l_address() {
  /* float (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof_l));
}


extern "C" intptr_t get_wcstold_l_address() {
  /* long double (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstold_l));
}


extern "C" intptr_t get_wcstof32_l_address() {
  /* _Float32 (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof32_l));
}


extern "C" intptr_t get_wcstof64_l_address() {
  /* _Float64 (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof64_l));
}


extern "C" intptr_t get_wcstof32x_l_address() {
  /* _Float32x (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof32x_l));
}


extern "C" intptr_t get_wcstof64x_l_address() {
  /* _Float64x (const wchar_t *__restrict, wchar_t **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstof64x_l));
}


extern "C" intptr_t get_wcpcpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcpcpy));
}


extern "C" intptr_t get_wcpncpy_address() {
  /* wchar_t *(wchar_t *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcpncpy));
}


extern "C" intptr_t get_open_wmemstream_address() {
  /* __FILE *(wchar_t **, size_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(open_wmemstream));
}


extern "C" intptr_t get_fwide_address() {
  /* int (__FILE *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fwide));
}


extern "C" intptr_t get_fwprintf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fwprintf));
}


extern "C" intptr_t get_wprintf_address() {
  /* int (const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(wprintf));
}


extern "C" intptr_t get_swprintf_address() {
  /* int (wchar_t *__restrict, size_t, const wchar_t *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(swprintf));
}


extern "C" intptr_t get_vfwprintf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfwprintf));
}


extern "C" intptr_t get_vwprintf_address() {
  /* int (const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vwprintf));
}


extern "C" intptr_t get_vswprintf_address() {
  /* int (wchar_t *__restrict, size_t, const wchar_t *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vswprintf));
}


extern "C" intptr_t get_fwscanf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fwscanf));
}


extern "C" intptr_t get_wscanf_address() {
  /* int (const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(wscanf));
}


extern "C" intptr_t get_swscanf_address() {
  /* int (const wchar_t *__restrict, const wchar_t *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(swscanf));
}


extern "C" intptr_t get_fwscanf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fwscanf));
}


extern "C" intptr_t get_wscanf_address() {
  /* int (const wchar_t *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(wscanf));
}


extern "C" intptr_t get_swscanf_address() {
  /* int (const wchar_t *__restrict, const wchar_t *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(swscanf));
}


extern "C" intptr_t get_vfwscanf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfwscanf));
}


extern "C" intptr_t get_vwscanf_address() {
  /* int (const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vwscanf));
}


extern "C" intptr_t get_vswscanf_address() {
  /* int (const wchar_t *__restrict, const wchar_t *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vswscanf));
}


extern "C" intptr_t get_vfwscanf_address() {
  /* int (__FILE *__restrict, const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfwscanf));
}


extern "C" intptr_t get_vwscanf_address() {
  /* int (const wchar_t *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vwscanf));
}


extern "C" intptr_t get_vswscanf_address() {
  /* int (const wchar_t *__restrict, const wchar_t *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vswscanf));
}


extern "C" intptr_t get_fgetwc_address() {
  /* wint_t (__FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetwc));
}


extern "C" intptr_t get_getwc_address() {
  /* wint_t (__FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(getwc));
}


extern "C" intptr_t get_getwchar_address() {
  /* wint_t () */
  return reinterpret_cast<intptr_t>(std::addressof(getwchar));
}


extern "C" intptr_t get_fputwc_address() {
  /* wint_t (wchar_t, __FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fputwc));
}


extern "C" intptr_t get_putwc_address() {
  /* wint_t (wchar_t, __FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(putwc));
}


extern "C" intptr_t get_putwchar_address() {
  /* wint_t (wchar_t) */
  return reinterpret_cast<intptr_t>(std::addressof(putwchar));
}


extern "C" intptr_t get_fgetws_address() {
  /* wchar_t *(wchar_t *__restrict, int, __FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetws));
}


extern "C" intptr_t get_fputws_address() {
  /* int (const wchar_t *__restrict, __FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fputws));
}


extern "C" intptr_t get_ungetwc_address() {
  /* wint_t (wint_t, __FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(ungetwc));
}


extern "C" intptr_t get_getwc_unlocked_address() {
  /* wint_t (__FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(getwc_unlocked));
}


extern "C" intptr_t get_getwchar_unlocked_address() {
  /* wint_t () */
  return reinterpret_cast<intptr_t>(std::addressof(getwchar_unlocked));
}


extern "C" intptr_t get_fgetwc_unlocked_address() {
  /* wint_t (__FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetwc_unlocked));
}


extern "C" intptr_t get_fputwc_unlocked_address() {
  /* wint_t (wchar_t, __FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fputwc_unlocked));
}


extern "C" intptr_t get_putwc_unlocked_address() {
  /* wint_t (wchar_t, __FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(putwc_unlocked));
}


extern "C" intptr_t get_putwchar_unlocked_address() {
  /* wint_t (wchar_t) */
  return reinterpret_cast<intptr_t>(std::addressof(putwchar_unlocked));
}


extern "C" intptr_t get_fgetws_unlocked_address() {
  /* wchar_t *(wchar_t *__restrict, int, __FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetws_unlocked));
}


extern "C" intptr_t get_fputws_unlocked_address() {
  /* int (const wchar_t *__restrict, __FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fputws_unlocked));
}


extern "C" intptr_t get_wcsftime_address() {
  /* size_t (wchar_t *__restrict, size_t, const wchar_t *__restrict, const struct tm *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsftime));
}


extern "C" intptr_t get_wcsftime_l_address() {
  /* size_t (wchar_t *__restrict, size_t, const wchar_t *__restrict, const struct tm *__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcsftime_l));
}


extern "C" intptr_t get_setlocale_address() {
  /* char *(int, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setlocale));
}


extern "C" intptr_t get_localeconv_address() {
  /* struct lconv *() noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(localeconv));
}


extern "C" intptr_t get_newlocale_address() {
  /* locale_t (int, const char *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(newlocale));
}


extern "C" intptr_t get_duplocale_address() {
  /* locale_t (locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(duplocale));
}


extern "C" intptr_t get_freelocale_address() {
  /* void (locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(freelocale));
}


extern "C" intptr_t get_uselocale_address() {
  /* locale_t (locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(uselocale));
}


extern "C" intptr_t get_isalnum_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isalnum));
}


extern "C" intptr_t get_isalpha_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isalpha));
}


extern "C" intptr_t get_iscntrl_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iscntrl));
}


extern "C" intptr_t get_isdigit_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isdigit));
}


extern "C" intptr_t get_islower_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(islower));
}


extern "C" intptr_t get_isgraph_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isgraph));
}


extern "C" intptr_t get_isprint_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isprint));
}


extern "C" intptr_t get_ispunct_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ispunct));
}


extern "C" intptr_t get_isspace_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isspace));
}


extern "C" intptr_t get_isupper_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isupper));
}


extern "C" intptr_t get_isxdigit_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isxdigit));
}


extern "C" intptr_t get_tolower_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tolower));
}


extern "C" intptr_t get_toupper_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(toupper));
}


extern "C" intptr_t get_isblank_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isblank));
}


extern "C" intptr_t get_isctype_address() {
  /* int (int, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isctype));
}


extern "C" intptr_t get_isascii_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isascii));
}


extern "C" intptr_t get_toascii_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(toascii));
}


extern "C" intptr_t get_isalnum_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isalnum_l));
}


extern "C" intptr_t get_isalpha_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isalpha_l));
}


extern "C" intptr_t get_iscntrl_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iscntrl_l));
}


extern "C" intptr_t get_isdigit_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isdigit_l));
}


extern "C" intptr_t get_islower_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(islower_l));
}


extern "C" intptr_t get_isgraph_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isgraph_l));
}


extern "C" intptr_t get_isprint_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isprint_l));
}


extern "C" intptr_t get_ispunct_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ispunct_l));
}


extern "C" intptr_t get_isspace_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isspace_l));
}


extern "C" intptr_t get_isupper_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isupper_l));
}


extern "C" intptr_t get_isxdigit_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isxdigit_l));
}


extern "C" intptr_t get_isblank_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(isblank_l));
}


extern "C" intptr_t get_tolower_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tolower_l));
}


extern "C" intptr_t get_toupper_l_address() {
  /* int (int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(toupper_l));
}


extern "C" intptr_t get_clone_address() {
  /* int (int (*)(void *), void *, int, void *, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clone));
}


extern "C" intptr_t get_unshare_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(unshare));
}


extern "C" intptr_t get_sched_getcpu_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_getcpu));
}


extern "C" intptr_t get_getcpu_address() {
  /* int (unsigned int *, unsigned int *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(getcpu));
}


extern "C" intptr_t get_setns_address() {
  /* int (int, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setns));
}


extern "C" intptr_t get_sched_setparam_address() {
  /* int (__pid_t, const struct sched_param *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_setparam));
}


extern "C" intptr_t get_sched_getparam_address() {
  /* int (__pid_t, struct sched_param *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_getparam));
}


extern "C" intptr_t get_sched_setscheduler_address() {
  /* int (__pid_t, int, const struct sched_param *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_setscheduler));
}


extern "C" intptr_t get_sched_getscheduler_address() {
  /* int (__pid_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_getscheduler));
}


extern "C" intptr_t get_sched_yield_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_yield));
}


extern "C" intptr_t get_sched_get_priority_max_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_get_priority_max));
}


extern "C" intptr_t get_sched_get_priority_min_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_get_priority_min));
}


extern "C" intptr_t get_sched_rr_get_interval_address() {
  /* int (__pid_t, struct timespec *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_rr_get_interval));
}


extern "C" intptr_t get_sched_setaffinity_address() {
  /* int (__pid_t, size_t, const cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_setaffinity));
}


extern "C" intptr_t get_sched_getaffinity_address() {
  /* int (__pid_t, size_t, cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sched_getaffinity));
}


extern "C" intptr_t get_clock_adjtime_address() {
  /* int (__clockid_t, struct timex *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_adjtime));
}


extern "C" intptr_t get_clock_address() {
  /* clock_t () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock));
}


extern "C" intptr_t get_time_address() {
  /* time_t (time_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(time));
}


extern "C" intptr_t get_difftime_address() {
  /* double (time_t, time_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(difftime));
}


extern "C" intptr_t get_mktime_address() {
  /* time_t (struct tm *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mktime));
}


extern "C" intptr_t get_strftime_address() {
  /* size_t (char *__restrict, size_t, const char *__restrict, const struct tm *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strftime));
}


extern "C" intptr_t get_strptime_address() {
  /* char *(const char *__restrict, const char *__restrict, struct tm *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strptime));
}


extern "C" intptr_t get_strftime_l_address() {
  /* size_t (char *__restrict, size_t, const char *__restrict, const struct tm *__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strftime_l));
}


extern "C" intptr_t get_strptime_l_address() {
  /* char *(const char *__restrict, const char *__restrict, struct tm *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strptime_l));
}


extern "C" intptr_t get_gmtime_address() {
  /* struct tm *(const time_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(gmtime));
}


extern "C" intptr_t get_localtime_address() {
  /* struct tm *(const time_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(localtime));
}


extern "C" intptr_t get_gmtime_r_address() {
  /* struct tm *(const time_t *__restrict, struct tm *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(gmtime_r));
}


extern "C" intptr_t get_localtime_r_address() {
  /* struct tm *(const time_t *__restrict, struct tm *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(localtime_r));
}


extern "C" intptr_t get_asctime_address() {
  /* char *(const struct tm *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(asctime));
}


extern "C" intptr_t get_ctime_address() {
  /* char *(const time_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ctime));
}


extern "C" intptr_t get_asctime_r_address() {
  /* char *(const struct tm *__restrict, char *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(asctime_r));
}


extern "C" intptr_t get_ctime_r_address() {
  /* char *(const time_t *__restrict, char *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ctime_r));
}


extern "C" intptr_t get_tzset_address() {
  /* void () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tzset));
}


extern "C" intptr_t get_timegm_address() {
  /* time_t (struct tm *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timegm));
}


extern "C" intptr_t get_timelocal_address() {
  /* time_t (struct tm *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timelocal));
}


extern "C" intptr_t get_dysize_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(dysize));
}


extern "C" intptr_t get_nanosleep_address() {
  /* int (const struct timespec *, struct timespec *) */
  return reinterpret_cast<intptr_t>(std::addressof(nanosleep));
}


extern "C" intptr_t get_clock_getres_address() {
  /* int (clockid_t, struct timespec *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_getres));
}


extern "C" intptr_t get_clock_gettime_address() {
  /* int (clockid_t, struct timespec *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_gettime));
}


extern "C" intptr_t get_clock_settime_address() {
  /* int (clockid_t, const struct timespec *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_settime));
}


extern "C" intptr_t get_clock_nanosleep_address() {
  /* int (clockid_t, int, const struct timespec *, struct timespec *) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_nanosleep));
}


extern "C" intptr_t get_clock_getcpuclockid_address() {
  /* int (pid_t, clockid_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clock_getcpuclockid));
}


extern "C" intptr_t get_timer_create_address() {
  /* int (clockid_t, struct sigevent *__restrict, timer_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timer_create));
}


extern "C" intptr_t get_timer_delete_address() {
  /* int (timer_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timer_delete));
}


extern "C" intptr_t get_timer_settime_address() {
  /* int (timer_t, int, const struct itimerspec *__restrict, struct itimerspec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timer_settime));
}


extern "C" intptr_t get_timer_gettime_address() {
  /* int (timer_t, struct itimerspec *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timer_gettime));
}


extern "C" intptr_t get_timer_getoverrun_address() {
  /* int (timer_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timer_getoverrun));
}


extern "C" intptr_t get_timespec_get_address() {
  /* int (struct timespec *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timespec_get));
}


extern "C" intptr_t get_timespec_getres_address() {
  /* int (struct timespec *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(timespec_getres));
}


extern "C" intptr_t get_getdate_address() {
  /* struct tm *(const char *) */
  return reinterpret_cast<intptr_t>(std::addressof(getdate));
}


extern "C" intptr_t get_getdate_r_address() {
  /* int (const char *__restrict, struct tm *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(getdate_r));
}


extern "C" intptr_t get_pthread_create_address() {
  /* int (pthread_t *__restrict, const pthread_attr_t *__restrict, void *(*)(void *), void *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_create));
}


extern "C" intptr_t get_pthread_exit_address() {
  /* void (void *) __attribute__((noreturn)) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_exit));
}


extern "C" intptr_t get_pthread_join_address() {
  /* int (pthread_t, void **) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_join));
}


extern "C" intptr_t get_pthread_tryjoin_np_address() {
  /* int (pthread_t, void **) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_tryjoin_np));
}


extern "C" intptr_t get_pthread_timedjoin_np_address() {
  /* int (pthread_t, void **, const struct timespec *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_timedjoin_np));
}


extern "C" intptr_t get_pthread_clockjoin_np_address() {
  /* int (pthread_t, void **, clockid_t, const struct timespec *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_clockjoin_np));
}


extern "C" intptr_t get_pthread_detach_address() {
  /* int (pthread_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_detach));
}


extern "C" intptr_t get_pthread_self_address() {
  /* pthread_t () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_self));
}


extern "C" intptr_t get_pthread_equal_address() {
  /* int (pthread_t, pthread_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_equal));
}


extern "C" intptr_t get_pthread_attr_init_address() {
  /* int (pthread_attr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_init));
}


extern "C" intptr_t get_pthread_attr_destroy_address() {
  /* int (pthread_attr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_destroy));
}


extern "C" intptr_t get_pthread_attr_getdetachstate_address() {
  /* int (const pthread_attr_t *, int *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getdetachstate));
}


extern "C" intptr_t get_pthread_attr_setdetachstate_address() {
  /* int (pthread_attr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setdetachstate));
}


extern "C" intptr_t get_pthread_attr_getguardsize_address() {
  /* int (const pthread_attr_t *, size_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getguardsize));
}


extern "C" intptr_t get_pthread_attr_setguardsize_address() {
  /* int (pthread_attr_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setguardsize));
}


extern "C" intptr_t get_pthread_attr_getschedparam_address() {
  /* int (const pthread_attr_t *__restrict, struct sched_param *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getschedparam));
}


extern "C" intptr_t get_pthread_attr_setschedparam_address() {
  /* int (pthread_attr_t *__restrict, const struct sched_param *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setschedparam));
}


extern "C" intptr_t get_pthread_attr_getschedpolicy_address() {
  /* int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getschedpolicy));
}


extern "C" intptr_t get_pthread_attr_setschedpolicy_address() {
  /* int (pthread_attr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setschedpolicy));
}


extern "C" intptr_t get_pthread_attr_getinheritsched_address() {
  /* int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getinheritsched));
}


extern "C" intptr_t get_pthread_attr_setinheritsched_address() {
  /* int (pthread_attr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setinheritsched));
}


extern "C" intptr_t get_pthread_attr_getscope_address() {
  /* int (const pthread_attr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getscope));
}


extern "C" intptr_t get_pthread_attr_setscope_address() {
  /* int (pthread_attr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setscope));
}


extern "C" intptr_t get_pthread_attr_getstackaddr_address() {
  /* int (const pthread_attr_t *__restrict, void **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getstackaddr));
}


extern "C" intptr_t get_pthread_attr_setstackaddr_address() {
  /* int (pthread_attr_t *, void *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setstackaddr));
}


extern "C" intptr_t get_pthread_attr_getstacksize_address() {
  /* int (const pthread_attr_t *__restrict, size_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getstacksize));
}


extern "C" intptr_t get_pthread_attr_setstacksize_address() {
  /* int (pthread_attr_t *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setstacksize));
}


extern "C" intptr_t get_pthread_attr_getstack_address() {
  /* int (const pthread_attr_t *__restrict, void **__restrict, size_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getstack));
}


extern "C" intptr_t get_pthread_attr_setstack_address() {
  /* int (pthread_attr_t *, void *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setstack));
}


extern "C" intptr_t get_pthread_attr_setaffinity_np_address() {
  /* int (pthread_attr_t *, size_t, const cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setaffinity_np));
}


extern "C" intptr_t get_pthread_attr_getaffinity_np_address() {
  /* int (const pthread_attr_t *, size_t, cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getaffinity_np));
}


extern "C" intptr_t get_pthread_getattr_default_np_address() {
  /* int (pthread_attr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getattr_default_np));
}


extern "C" intptr_t get_pthread_attr_setsigmask_np_address() {
  /* int (pthread_attr_t *, const __sigset_t *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_setsigmask_np));
}


extern "C" intptr_t get_pthread_attr_getsigmask_np_address() {
  /* int (const pthread_attr_t *, __sigset_t *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_attr_getsigmask_np));
}


extern "C" intptr_t get_pthread_setattr_default_np_address() {
  /* int (const pthread_attr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setattr_default_np));
}


extern "C" intptr_t get_pthread_getattr_np_address() {
  /* int (pthread_t, pthread_attr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getattr_np));
}


extern "C" intptr_t get_pthread_setschedparam_address() {
  /* int (pthread_t, int, const struct sched_param *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setschedparam));
}


extern "C" intptr_t get_pthread_getschedparam_address() {
  /* int (pthread_t, int *__restrict, struct sched_param *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getschedparam));
}


extern "C" intptr_t get_pthread_setschedprio_address() {
  /* int (pthread_t, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setschedprio));
}


extern "C" intptr_t get_pthread_getname_np_address() {
  /* int (pthread_t, char *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getname_np));
}


extern "C" intptr_t get_pthread_setname_np_address() {
  /* int (pthread_t, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setname_np));
}


extern "C" intptr_t get_pthread_getconcurrency_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getconcurrency));
}


extern "C" intptr_t get_pthread_setconcurrency_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setconcurrency));
}


extern "C" intptr_t get_pthread_yield_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_yield));
}


extern "C" intptr_t get_pthread_yield_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_yield));
}


extern "C" intptr_t get_pthread_setaffinity_np_address() {
  /* int (pthread_t, size_t, const cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setaffinity_np));
}


extern "C" intptr_t get_pthread_getaffinity_np_address() {
  /* int (pthread_t, size_t, cpu_set_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getaffinity_np));
}


extern "C" intptr_t get_pthread_once_address() {
  /* int (pthread_once_t *, void (*)()) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_once));
}


extern "C" intptr_t get_pthread_setcancelstate_address() {
  /* int (int, int *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setcancelstate));
}


extern "C" intptr_t get_pthread_setcanceltype_address() {
  /* int (int, int *) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setcanceltype));
}


extern "C" intptr_t get_pthread_cancel_address() {
  /* int (pthread_t) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cancel));
}


extern "C" intptr_t get_pthread_testcancel_address() {
  /* void () */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_testcancel));
}


extern "C" intptr_t get_pthread_mutex_init_address() {
  /* int (pthread_mutex_t *, const pthread_mutexattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_init));
}


extern "C" intptr_t get_pthread_mutex_destroy_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_destroy));
}


extern "C" intptr_t get_pthread_mutex_trylock_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_trylock));
}


extern "C" intptr_t get_pthread_mutex_lock_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_lock));
}


extern "C" intptr_t get_pthread_mutex_timedlock_address() {
  /* int (pthread_mutex_t *__restrict, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_timedlock));
}


extern "C" intptr_t get_pthread_mutex_clocklock_address() {
  /* int (pthread_mutex_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_clocklock));
}


extern "C" intptr_t get_pthread_mutex_unlock_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_unlock));
}


extern "C" intptr_t get_pthread_mutex_getprioceiling_address() {
  /* int (const pthread_mutex_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_getprioceiling));
}


extern "C" intptr_t get_pthread_mutex_setprioceiling_address() {
  /* int (pthread_mutex_t *__restrict, int, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_setprioceiling));
}


extern "C" intptr_t get_pthread_mutex_consistent_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_consistent));
}


extern "C" intptr_t get_pthread_mutex_consistent_np_address() {
  /* int (pthread_mutex_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutex_consistent_np));
}


extern "C" intptr_t get_pthread_mutexattr_init_address() {
  /* int (pthread_mutexattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_init));
}


extern "C" intptr_t get_pthread_mutexattr_destroy_address() {
  /* int (pthread_mutexattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_destroy));
}


extern "C" intptr_t get_pthread_mutexattr_getpshared_address() {
  /* int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_getpshared));
}


extern "C" intptr_t get_pthread_mutexattr_setpshared_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_setpshared));
}


extern "C" intptr_t get_pthread_mutexattr_gettype_address() {
  /* int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_gettype));
}


extern "C" intptr_t get_pthread_mutexattr_settype_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_settype));
}


extern "C" intptr_t get_pthread_mutexattr_getprotocol_address() {
  /* int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_getprotocol));
}


extern "C" intptr_t get_pthread_mutexattr_setprotocol_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_setprotocol));
}


extern "C" intptr_t get_pthread_mutexattr_getprioceiling_address() {
  /* int (const pthread_mutexattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_getprioceiling));
}


extern "C" intptr_t get_pthread_mutexattr_setprioceiling_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_setprioceiling));
}


extern "C" intptr_t get_pthread_mutexattr_getrobust_address() {
  /* int (const pthread_mutexattr_t *, int *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_getrobust));
}


extern "C" intptr_t get_pthread_mutexattr_getrobust_np_address() {
  /* int (pthread_mutexattr_t *, int *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_getrobust_np));
}


extern "C" intptr_t get_pthread_mutexattr_setrobust_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_setrobust));
}


extern "C" intptr_t get_pthread_mutexattr_setrobust_np_address() {
  /* int (pthread_mutexattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_mutexattr_setrobust_np));
}


extern "C" intptr_t get_pthread_rwlock_init_address() {
  /* int (pthread_rwlock_t *__restrict, const pthread_rwlockattr_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_init));
}


extern "C" intptr_t get_pthread_rwlock_destroy_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_destroy));
}


extern "C" intptr_t get_pthread_rwlock_rdlock_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_rdlock));
}


extern "C" intptr_t get_pthread_rwlock_tryrdlock_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_tryrdlock));
}


extern "C" intptr_t get_pthread_rwlock_timedrdlock_address() {
  /* int (pthread_rwlock_t *__restrict, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_timedrdlock));
}


extern "C" intptr_t get_pthread_rwlock_clockrdlock_address() {
  /* int (pthread_rwlock_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_clockrdlock));
}


extern "C" intptr_t get_pthread_rwlock_wrlock_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_wrlock));
}


extern "C" intptr_t get_pthread_rwlock_trywrlock_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_trywrlock));
}


extern "C" intptr_t get_pthread_rwlock_timedwrlock_address() {
  /* int (pthread_rwlock_t *__restrict, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_timedwrlock));
}


extern "C" intptr_t get_pthread_rwlock_clockwrlock_address() {
  /* int (pthread_rwlock_t *__restrict, clockid_t, const struct timespec *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_clockwrlock));
}


extern "C" intptr_t get_pthread_rwlock_unlock_address() {
  /* int (pthread_rwlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlock_unlock));
}


extern "C" intptr_t get_pthread_rwlockattr_init_address() {
  /* int (pthread_rwlockattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_init));
}


extern "C" intptr_t get_pthread_rwlockattr_destroy_address() {
  /* int (pthread_rwlockattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_destroy));
}


extern "C" intptr_t get_pthread_rwlockattr_getpshared_address() {
  /* int (const pthread_rwlockattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_getpshared));
}


extern "C" intptr_t get_pthread_rwlockattr_setpshared_address() {
  /* int (pthread_rwlockattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_setpshared));
}


extern "C" intptr_t get_pthread_rwlockattr_getkind_np_address() {
  /* int (const pthread_rwlockattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_getkind_np));
}


extern "C" intptr_t get_pthread_rwlockattr_setkind_np_address() {
  /* int (pthread_rwlockattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_rwlockattr_setkind_np));
}


extern "C" intptr_t get_pthread_cond_init_address() {
  /* int (pthread_cond_t *__restrict, const pthread_condattr_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_init));
}


extern "C" intptr_t get_pthread_cond_destroy_address() {
  /* int (pthread_cond_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_destroy));
}


extern "C" intptr_t get_pthread_cond_signal_address() {
  /* int (pthread_cond_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_signal));
}


extern "C" intptr_t get_pthread_cond_broadcast_address() {
  /* int (pthread_cond_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_broadcast));
}


extern "C" intptr_t get_pthread_cond_wait_address() {
  /* int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_wait));
}


extern "C" intptr_t get_pthread_cond_timedwait_address() {
  /* int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict, const struct timespec *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_timedwait));
}


extern "C" intptr_t get_pthread_cond_clockwait_address() {
  /* int (pthread_cond_t *__restrict, pthread_mutex_t *__restrict, __clockid_t, const struct timespec *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_cond_clockwait));
}


extern "C" intptr_t get_pthread_condattr_init_address() {
  /* int (pthread_condattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_init));
}


extern "C" intptr_t get_pthread_condattr_destroy_address() {
  /* int (pthread_condattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_destroy));
}


extern "C" intptr_t get_pthread_condattr_getpshared_address() {
  /* int (const pthread_condattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_getpshared));
}


extern "C" intptr_t get_pthread_condattr_setpshared_address() {
  /* int (pthread_condattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_setpshared));
}


extern "C" intptr_t get_pthread_condattr_getclock_address() {
  /* int (const pthread_condattr_t *__restrict, __clockid_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_getclock));
}


extern "C" intptr_t get_pthread_condattr_setclock_address() {
  /* int (pthread_condattr_t *, __clockid_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_condattr_setclock));
}


extern "C" intptr_t get_pthread_spin_init_address() {
  /* int (pthread_spinlock_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_spin_init));
}


extern "C" intptr_t get_pthread_spin_destroy_address() {
  /* int (pthread_spinlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_spin_destroy));
}


extern "C" intptr_t get_pthread_spin_lock_address() {
  /* int (pthread_spinlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_spin_lock));
}


extern "C" intptr_t get_pthread_spin_trylock_address() {
  /* int (pthread_spinlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_spin_trylock));
}


extern "C" intptr_t get_pthread_spin_unlock_address() {
  /* int (pthread_spinlock_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_spin_unlock));
}


extern "C" intptr_t get_pthread_barrier_init_address() {
  /* int (pthread_barrier_t *__restrict, const pthread_barrierattr_t *__restrict, unsigned int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrier_init));
}


extern "C" intptr_t get_pthread_barrier_destroy_address() {
  /* int (pthread_barrier_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrier_destroy));
}


extern "C" intptr_t get_pthread_barrier_wait_address() {
  /* int (pthread_barrier_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrier_wait));
}


extern "C" intptr_t get_pthread_barrierattr_init_address() {
  /* int (pthread_barrierattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrierattr_init));
}


extern "C" intptr_t get_pthread_barrierattr_destroy_address() {
  /* int (pthread_barrierattr_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrierattr_destroy));
}


extern "C" intptr_t get_pthread_barrierattr_getpshared_address() {
  /* int (const pthread_barrierattr_t *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrierattr_getpshared));
}


extern "C" intptr_t get_pthread_barrierattr_setpshared_address() {
  /* int (pthread_barrierattr_t *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_barrierattr_setpshared));
}


extern "C" intptr_t get_pthread_key_create_address() {
  /* int (pthread_key_t *, void (*)(void *)) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_key_create));
}


extern "C" intptr_t get_pthread_key_delete_address() {
  /* int (pthread_key_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_key_delete));
}


extern "C" intptr_t get_pthread_getspecific_address() {
  /* void *(pthread_key_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getspecific));
}


extern "C" intptr_t get_pthread_setspecific_address() {
  /* int (pthread_key_t, const void *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_setspecific));
}


extern "C" intptr_t get_pthread_getcpuclockid_address() {
  /* int (pthread_t, __clockid_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_getcpuclockid));
}


extern "C" intptr_t get_pthread_atfork_address() {
  /* int (void (*)(), void (*)(), void (*)()) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(pthread_atfork));
}


extern "C" intptr_t get_atof_address() {
  /* double (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(atof));
}


extern "C" intptr_t get_atoi_address() {
  /* int (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(atoi));
}


extern "C" intptr_t get_atol_address() {
  /* long (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(atol));
}


extern "C" intptr_t get_atoll_address() {
  /* long long (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(atoll));
}


extern "C" intptr_t get_strtod_address() {
  /* double (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtod));
}


extern "C" intptr_t get_strtof_address() {
  /* float (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof));
}


extern "C" intptr_t get_strtold_address() {
  /* long double (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtold));
}


extern "C" intptr_t get_strtof32_address() {
  /* _Float32 (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof32));
}


extern "C" intptr_t get_strtof64_address() {
  /* _Float64 (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof64));
}


extern "C" intptr_t get_strtof32x_address() {
  /* _Float32x (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof32x));
}


extern "C" intptr_t get_strtof64x_address() {
  /* _Float64x (const char *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof64x));
}


extern "C" intptr_t get_strtol_address() {
  /* long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtol));
}


extern "C" intptr_t get_strtoul_address() {
  /* unsigned long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoul));
}


extern "C" intptr_t get_strtoq_address() {
  /* long long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoq));
}


extern "C" intptr_t get_strtouq_address() {
  /* unsigned long long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtouq));
}


extern "C" intptr_t get_strtoll_address() {
  /* long long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoll));
}


extern "C" intptr_t get_strtoull_address() {
  /* unsigned long long (const char *__restrict, char **__restrict, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoull));
}


extern "C" intptr_t get_strfromd_address() {
  /* int (char *, size_t, const char *, double) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromd));
}


extern "C" intptr_t get_strfromf_address() {
  /* int (char *, size_t, const char *, float) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromf));
}


extern "C" intptr_t get_strfroml_address() {
  /* int (char *, size_t, const char *, long double) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfroml));
}


extern "C" intptr_t get_strfromf32_address() {
  /* int (char *, size_t, const char *, _Float32) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromf32));
}


extern "C" intptr_t get_strfromf64_address() {
  /* int (char *, size_t, const char *, _Float64) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromf64));
}


extern "C" intptr_t get_strfromf32x_address() {
  /* int (char *, size_t, const char *, _Float32x) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromf32x));
}


extern "C" intptr_t get_strfromf64x_address() {
  /* int (char *, size_t, const char *, _Float64x) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strfromf64x));
}


extern "C" intptr_t get_strtol_l_address() {
  /* long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtol_l));
}


extern "C" intptr_t get_strtoul_l_address() {
  /* unsigned long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoul_l));
}


extern "C" intptr_t get_strtoll_l_address() {
  /* long long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoll_l));
}


extern "C" intptr_t get_strtoull_l_address() {
  /* unsigned long long (const char *__restrict, char **__restrict, int, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtoull_l));
}


extern "C" intptr_t get_strtod_l_address() {
  /* double (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtod_l));
}


extern "C" intptr_t get_strtof_l_address() {
  /* float (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof_l));
}


extern "C" intptr_t get_strtold_l_address() {
  /* long double (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtold_l));
}


extern "C" intptr_t get_strtof32_l_address() {
  /* _Float32 (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof32_l));
}


extern "C" intptr_t get_strtof64_l_address() {
  /* _Float64 (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof64_l));
}


extern "C" intptr_t get_strtof32x_l_address() {
  /* _Float32x (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof32x_l));
}


extern "C" intptr_t get_strtof64x_l_address() {
  /* _Float64x (const char *__restrict, char **__restrict, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(strtof64x_l));
}


extern "C" intptr_t get_l64a_address() {
  /* char *(long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(l64a));
}


extern "C" intptr_t get_a64l_address() {
  /* long (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(a64l));
}


extern "C" intptr_t get_select_address() {
  /* int (int, fd_set *__restrict, fd_set *__restrict, fd_set *__restrict, struct timeval *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(select));
}


extern "C" intptr_t get_pselect_address() {
  /* int (int, fd_set *__restrict, fd_set *__restrict, fd_set *__restrict, const struct timespec *__restrict, const __sigset_t *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(pselect));
}


extern "C" intptr_t get_random_address() {
  /* long () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(random));
}


extern "C" intptr_t get_srandom_address() {
  /* void (unsigned int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(srandom));
}


extern "C" intptr_t get_initstate_address() {
  /* char *(unsigned int, char *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(initstate));
}


extern "C" intptr_t get_setstate_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setstate));
}


extern "C" intptr_t get_random_r_address() {
  /* int (struct random_data *__restrict, int32_t *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(random_r));
}


extern "C" intptr_t get_srandom_r_address() {
  /* int (unsigned int, struct random_data *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(srandom_r));
}


extern "C" intptr_t get_initstate_r_address() {
  /* int (unsigned int, char *__restrict, size_t, struct random_data *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(initstate_r));
}


extern "C" intptr_t get_setstate_r_address() {
  /* int (char *__restrict, struct random_data *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setstate_r));
}


extern "C" intptr_t get_rand_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(rand));
}


extern "C" intptr_t get_srand_address() {
  /* void (unsigned int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(srand));
}


extern "C" intptr_t get_rand_r_address() {
  /* int (unsigned int *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(rand_r));
}


extern "C" intptr_t get_drand48_address() {
  /* double () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(drand48));
}


extern "C" intptr_t get_erand48_address() {
  /* double (unsigned short *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(erand48));
}


extern "C" intptr_t get_lrand48_address() {
  /* long () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(lrand48));
}


extern "C" intptr_t get_nrand48_address() {
  /* long (unsigned short *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(nrand48));
}


extern "C" intptr_t get_mrand48_address() {
  /* long () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mrand48));
}


extern "C" intptr_t get_jrand48_address() {
  /* long (unsigned short *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(jrand48));
}


extern "C" intptr_t get_srand48_address() {
  /* void (long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(srand48));
}


extern "C" intptr_t get_seed48_address() {
  /* unsigned short *(unsigned short *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(seed48));
}


extern "C" intptr_t get_lcong48_address() {
  /* void (unsigned short *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(lcong48));
}


extern "C" intptr_t get_drand48_r_address() {
  /* int (struct drand48_data *__restrict, double *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(drand48_r));
}


extern "C" intptr_t get_erand48_r_address() {
  /* int (unsigned short *, struct drand48_data *__restrict, double *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(erand48_r));
}


extern "C" intptr_t get_lrand48_r_address() {
  /* int (struct drand48_data *__restrict, long *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(lrand48_r));
}


extern "C" intptr_t get_nrand48_r_address() {
  /* int (unsigned short *, struct drand48_data *__restrict, long *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(nrand48_r));
}


extern "C" intptr_t get_mrand48_r_address() {
  /* int (struct drand48_data *__restrict, long *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mrand48_r));
}


extern "C" intptr_t get_jrand48_r_address() {
  /* int (unsigned short *, struct drand48_data *__restrict, long *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(jrand48_r));
}


extern "C" intptr_t get_srand48_r_address() {
  /* int (long, struct drand48_data *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(srand48_r));
}


extern "C" intptr_t get_seed48_r_address() {
  /* int (unsigned short *, struct drand48_data *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(seed48_r));
}


extern "C" intptr_t get_lcong48_r_address() {
  /* int (unsigned short *, struct drand48_data *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(lcong48_r));
}


extern "C" intptr_t get_malloc_address() {
  /* void *(size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(malloc));
}


extern "C" intptr_t get_calloc_address() {
  /* void *(size_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(calloc));
}


extern "C" intptr_t get_realloc_address() {
  /* void *(void *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(realloc));
}


extern "C" intptr_t get_free_address() {
  /* void (void *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(free));
}


extern "C" intptr_t get_reallocarray_address() {
  /* void *(void *, size_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(reallocarray));
}


extern "C" intptr_t get_reallocarray_address() {
  /* void *(void *, size_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(reallocarray));
}


extern "C" intptr_t get_alloca_address() {
  /* void *(size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(alloca));
}


extern "C" intptr_t get_valloc_address() {
  /* void *(size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(valloc));
}


extern "C" intptr_t get_posix_memalign_address() {
  /* int (void **, size_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(posix_memalign));
}


extern "C" intptr_t get_aligned_alloc_address() {
  /* void *(size_t, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(aligned_alloc));
}


extern "C" intptr_t get_abort_address() {
  /* void () __attribute__((noreturn)) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(abort));
}


extern "C" intptr_t get_atexit_address() {
  /* int (void (*)()) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(atexit));
}


extern "C" intptr_t get_at_quick_exit_address() {
  /* int (void (*)()) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(at_quick_exit));
}


extern "C" intptr_t get_on_exit_address() {
  /* int (void (*)(int, void *), void *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(on_exit));
}


extern "C" intptr_t get_exit_address() {
  /* void (int) __attribute__((noreturn)) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(exit));
}


extern "C" intptr_t get_quick_exit_address() {
  /* void (int) __attribute__((noreturn)) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(quick_exit));
}


extern "C" intptr_t get_getenv_address() {
  /* char *(const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(getenv));
}


extern "C" intptr_t get_secure_getenv_address() {
  /* char *(const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(secure_getenv));
}


extern "C" intptr_t get_putenv_address() {
  /* int (char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(putenv));
}


extern "C" intptr_t get_setenv_address() {
  /* int (const char *, const char *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setenv));
}


extern "C" intptr_t get_unsetenv_address() {
  /* int (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(unsetenv));
}


extern "C" intptr_t get_clearenv_address() {
  /* int () noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clearenv));
}


extern "C" intptr_t get_mktemp_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mktemp));
}


extern "C" intptr_t get_mkstemp_address() {
  /* int (char *) */
  return reinterpret_cast<intptr_t>(std::addressof(mkstemp));
}


extern "C" intptr_t get_mkstemp64_address() {
  /* int (char *) */
  return reinterpret_cast<intptr_t>(std::addressof(mkstemp64));
}


extern "C" intptr_t get_mkstemps_address() {
  /* int (char *, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkstemps));
}


extern "C" intptr_t get_mkstemps64_address() {
  /* int (char *, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkstemps64));
}


extern "C" intptr_t get_mkdtemp_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mkdtemp));
}


extern "C" intptr_t get_mkostemp_address() {
  /* int (char *, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkostemp));
}


extern "C" intptr_t get_mkostemp64_address() {
  /* int (char *, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkostemp64));
}


extern "C" intptr_t get_mkostemps_address() {
  /* int (char *, int, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkostemps));
}


extern "C" intptr_t get_mkostemps64_address() {
  /* int (char *, int, int) */
  return reinterpret_cast<intptr_t>(std::addressof(mkostemps64));
}


extern "C" intptr_t get_system_address() {
  /* int (const char *) */
  return reinterpret_cast<intptr_t>(std::addressof(system));
}


extern "C" intptr_t get_canonicalize_file_name_address() {
  /* char *(const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(canonicalize_file_name));
}


extern "C" intptr_t get_realpath_address() {
  /* char *(const char *__restrict, char *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(realpath));
}


extern "C" intptr_t get_bsearch_address() {
  /* void *(const void *, const void *, size_t, size_t, __compar_fn_t) */
  return reinterpret_cast<intptr_t>(std::addressof(bsearch));
}


extern "C" intptr_t get_qsort_address() {
  /* void (void *, size_t, size_t, __compar_fn_t) */
  return reinterpret_cast<intptr_t>(std::addressof(qsort));
}


extern "C" intptr_t get_qsort_r_address() {
  /* void (void *, size_t, size_t, __compar_d_fn_t, void *) */
  return reinterpret_cast<intptr_t>(std::addressof(qsort_r));
}


extern "C" intptr_t get_abs_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(abs));
}


extern "C" intptr_t get_labs_address() {
  /* long (long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(labs));
}


extern "C" intptr_t get_llabs_address() {
  /* long long (long long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(llabs));
}


extern "C" intptr_t get_div_address() {
  /* div_t (int, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(div));
}


extern "C" intptr_t get_ldiv_address() {
  /* ldiv_t (long, long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ldiv));
}


extern "C" intptr_t get_lldiv_address() {
  /* lldiv_t (long long, long long) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(lldiv));
}


extern "C" intptr_t get_ecvt_address() {
  /* char *(double, int, int *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ecvt));
}


extern "C" intptr_t get_fcvt_address() {
  /* char *(double, int, int *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fcvt));
}


extern "C" intptr_t get_gcvt_address() {
  /* char *(double, int, char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(gcvt));
}


extern "C" intptr_t get_qecvt_address() {
  /* char *(long double, int, int *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(qecvt));
}


extern "C" intptr_t get_qfcvt_address() {
  /* char *(long double, int, int *__restrict, int *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(qfcvt));
}


extern "C" intptr_t get_qgcvt_address() {
  /* char *(long double, int, char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(qgcvt));
}


extern "C" intptr_t get_ecvt_r_address() {
  /* int (double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ecvt_r));
}


extern "C" intptr_t get_fcvt_r_address() {
  /* int (double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fcvt_r));
}


extern "C" intptr_t get_qecvt_r_address() {
  /* int (long double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(qecvt_r));
}


extern "C" intptr_t get_qfcvt_r_address() {
  /* int (long double, int, int *__restrict, int *__restrict, char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(qfcvt_r));
}


extern "C" intptr_t get_mblen_address() {
  /* int (const char *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mblen));
}


extern "C" intptr_t get_mbtowc_address() {
  /* int (wchar_t *__restrict, const char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbtowc));
}


extern "C" intptr_t get_wctomb_address() {
  /* int (char *, wchar_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctomb));
}


extern "C" intptr_t get_mbstowcs_address() {
  /* size_t (wchar_t *__restrict, const char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(mbstowcs));
}


extern "C" intptr_t get_wcstombs_address() {
  /* size_t (char *__restrict, const wchar_t *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wcstombs));
}


extern "C" intptr_t get_rpmatch_address() {
  /* int (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(rpmatch));
}


extern "C" intptr_t get_getsubopt_address() {
  /* int (char **__restrict, char *const *__restrict, char **__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(getsubopt));
}


extern "C" intptr_t get_posix_openpt_address() {
  /* int (int) */
  return reinterpret_cast<intptr_t>(std::addressof(posix_openpt));
}


extern "C" intptr_t get_grantpt_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(grantpt));
}


extern "C" intptr_t get_unlockpt_address() {
  /* int (int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(unlockpt));
}


extern "C" intptr_t get_ptsname_address() {
  /* char *(int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ptsname));
}


extern "C" intptr_t get_ptsname_r_address() {
  /* int (int, char *, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ptsname_r));
}


extern "C" intptr_t get_getpt_address() {
  /* int () */
  return reinterpret_cast<intptr_t>(std::addressof(getpt));
}


extern "C" intptr_t get_getloadavg_address() {
  /* int (double *, int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(getloadavg));
}


extern "C" intptr_t get_remove_address() {
  /* int (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(remove));
}


extern "C" intptr_t get_rename_address() {
  /* int (const char *, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(rename));
}


extern "C" intptr_t get_renameat_address() {
  /* int (int, const char *, int, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(renameat));
}


extern "C" intptr_t get_renameat2_address() {
  /* int (int, const char *, int, const char *, unsigned int) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(renameat2));
}


extern "C" intptr_t get_fclose_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fclose));
}


extern "C" intptr_t get_tmpfile_address() {
  /* FILE *() */
  return reinterpret_cast<intptr_t>(std::addressof(tmpfile));
}


extern "C" intptr_t get_tmpfile64_address() {
  /* FILE *() */
  return reinterpret_cast<intptr_t>(std::addressof(tmpfile64));
}


extern "C" intptr_t get_tmpnam_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tmpnam));
}


extern "C" intptr_t get_tmpnam_r_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tmpnam_r));
}


extern "C" intptr_t get_tempnam_address() {
  /* char *(const char *, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(tempnam));
}


extern "C" intptr_t get_fflush_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fflush));
}


extern "C" intptr_t get_fflush_unlocked_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fflush_unlocked));
}


extern "C" intptr_t get_fcloseall_address() {
  /* int () */
  return reinterpret_cast<intptr_t>(std::addressof(fcloseall));
}


extern "C" intptr_t get_fopen_address() {
  /* FILE *(const char *__restrict, const char *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fopen));
}


extern "C" intptr_t get_freopen_address() {
  /* FILE *(const char *__restrict, const char *__restrict, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(freopen));
}


extern "C" intptr_t get_fopen64_address() {
  /* FILE *(const char *__restrict, const char *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fopen64));
}


extern "C" intptr_t get_freopen64_address() {
  /* FILE *(const char *__restrict, const char *__restrict, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(freopen64));
}


extern "C" intptr_t get_fdopen_address() {
  /* FILE *(int, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fdopen));
}


extern "C" intptr_t get_fopencookie_address() {
  /* FILE *(void *__restrict, const char *__restrict, cookie_io_functions_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fopencookie));
}


extern "C" intptr_t get_fmemopen_address() {
  /* FILE *(void *, size_t, const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fmemopen));
}


extern "C" intptr_t get_open_memstream_address() {
  /* FILE *(char **, size_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(open_memstream));
}


extern "C" intptr_t get_open_wmemstream_address() {
  /* __FILE *(wchar_t **, size_t *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(open_wmemstream));
}


extern "C" intptr_t get_setbuf_address() {
  /* void (FILE *__restrict, char *__restrict) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setbuf));
}


extern "C" intptr_t get_setvbuf_address() {
  /* int (FILE *__restrict, char *__restrict, int, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setvbuf));
}


extern "C" intptr_t get_setbuffer_address() {
  /* void (FILE *__restrict, char *__restrict, size_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setbuffer));
}


extern "C" intptr_t get_setlinebuf_address() {
  /* void (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(setlinebuf));
}


extern "C" intptr_t get_fprintf_address() {
  /* int (FILE *__restrict, const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fprintf));
}


extern "C" intptr_t get_printf_address() {
  /* int (const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(printf));
}


extern "C" intptr_t get_sprintf_address() {
  /* int (char *__restrict, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sprintf));
}


extern "C" intptr_t get_vfprintf_address() {
  /* int (FILE *__restrict, const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfprintf));
}


extern "C" intptr_t get_vprintf_address() {
  /* int (const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vprintf));
}


extern "C" intptr_t get_vsprintf_address() {
  /* int (char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vsprintf));
}


extern "C" intptr_t get_snprintf_address() {
  /* int (char *__restrict, size_t, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(snprintf));
}


extern "C" intptr_t get_vsnprintf_address() {
  /* int (char *__restrict, size_t, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vsnprintf));
}


extern "C" intptr_t get_vasprintf_address() {
  /* int (char **__restrict, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vasprintf));
}


extern "C" intptr_t get_asprintf_address() {
  /* int (char **__restrict, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(asprintf));
}


extern "C" intptr_t get_vdprintf_address() {
  /* int (int, const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vdprintf));
}


extern "C" intptr_t get_dprintf_address() {
  /* int (int, const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(dprintf));
}


extern "C" intptr_t get_fscanf_address() {
  /* int (FILE *__restrict, const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fscanf));
}


extern "C" intptr_t get_scanf_address() {
  /* int (const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(scanf));
}


extern "C" intptr_t get_sscanf_address() {
  /* int (const char *__restrict, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sscanf));
}


extern "C" intptr_t get_fscanf_address() {
  /* int (FILE *__restrict, const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(fscanf));
}


extern "C" intptr_t get_scanf_address() {
  /* int (const char *__restrict, ...) */
  return reinterpret_cast<intptr_t>(std::addressof(scanf));
}


extern "C" intptr_t get_sscanf_address() {
  /* int (const char *__restrict, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(sscanf));
}


extern "C" intptr_t get_vfscanf_address() {
  /* int (FILE *__restrict, const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfscanf));
}


extern "C" intptr_t get_vscanf_address() {
  /* int (const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vscanf));
}


extern "C" intptr_t get_vsscanf_address() {
  /* int (const char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vsscanf));
}


extern "C" intptr_t get_vfscanf_address() {
  /* int (FILE *__restrict, const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vfscanf));
}


extern "C" intptr_t get_vscanf_address() {
  /* int (const char *__restrict, __va_list_tag *) */
  return reinterpret_cast<intptr_t>(std::addressof(vscanf));
}


extern "C" intptr_t get_vsscanf_address() {
  /* int (const char *__restrict, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(vsscanf));
}


extern "C" intptr_t get_fgetc_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetc));
}


extern "C" intptr_t get_getc_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(getc));
}


extern "C" intptr_t get_getchar_address() {
  /* int () */
  return reinterpret_cast<intptr_t>(std::addressof(getchar));
}


extern "C" intptr_t get_getc_unlocked_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(getc_unlocked));
}


extern "C" intptr_t get_getchar_unlocked_address() {
  /* int () */
  return reinterpret_cast<intptr_t>(std::addressof(getchar_unlocked));
}


extern "C" intptr_t get_fgetc_unlocked_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetc_unlocked));
}


extern "C" intptr_t get_fputc_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fputc));
}


extern "C" intptr_t get_putc_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(putc));
}


extern "C" intptr_t get_putchar_address() {
  /* int (int) */
  return reinterpret_cast<intptr_t>(std::addressof(putchar));
}


extern "C" intptr_t get_fputc_unlocked_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(fputc_unlocked));
}


extern "C" intptr_t get_putc_unlocked_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(putc_unlocked));
}


extern "C" intptr_t get_putchar_unlocked_address() {
  /* int (int) */
  return reinterpret_cast<intptr_t>(std::addressof(putchar_unlocked));
}


extern "C" intptr_t get_getw_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(getw));
}


extern "C" intptr_t get_putw_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(putw));
}


extern "C" intptr_t get_fgets_address() {
  /* char *(char *__restrict, int, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgets));
}


extern "C" intptr_t get_fgets_unlocked_address() {
  /* char *(char *__restrict, int, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgets_unlocked));
}


extern "C" intptr_t get_getdelim_address() {
  /* __ssize_t (char **__restrict, size_t *__restrict, int, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(getdelim));
}


extern "C" intptr_t get_getline_address() {
  /* __ssize_t (char **__restrict, size_t *__restrict, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(getline));
}


extern "C" intptr_t get_fputs_address() {
  /* int (const char *__restrict, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fputs));
}


extern "C" intptr_t get_puts_address() {
  /* int (const char *) */
  return reinterpret_cast<intptr_t>(std::addressof(puts));
}


extern "C" intptr_t get_ungetc_address() {
  /* int (int, FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(ungetc));
}


extern "C" intptr_t get_fread_address() {
  /* size_t (void *__restrict, size_t, size_t, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fread));
}


extern "C" intptr_t get_fwrite_address() {
  /* size_t (const void *__restrict, size_t, size_t, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fwrite));
}


extern "C" intptr_t get_fputs_unlocked_address() {
  /* int (const char *__restrict, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fputs_unlocked));
}


extern "C" intptr_t get_fread_unlocked_address() {
  /* size_t (void *__restrict, size_t, size_t, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fread_unlocked));
}


extern "C" intptr_t get_fwrite_unlocked_address() {
  /* size_t (const void *__restrict, size_t, size_t, FILE *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fwrite_unlocked));
}


extern "C" intptr_t get_fseek_address() {
  /* int (FILE *, long, int) */
  return reinterpret_cast<intptr_t>(std::addressof(fseek));
}


extern "C" intptr_t get_ftell_address() {
  /* long (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(ftell));
}


extern "C" intptr_t get_rewind_address() {
  /* void (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(rewind));
}


extern "C" intptr_t get_fseeko_address() {
  /* int (FILE *, __off_t, int) */
  return reinterpret_cast<intptr_t>(std::addressof(fseeko));
}


extern "C" intptr_t get_ftello_address() {
  /* __off_t (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(ftello));
}


extern "C" intptr_t get_fgetpos_address() {
  /* int (FILE *__restrict, fpos_t *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetpos));
}


extern "C" intptr_t get_fsetpos_address() {
  /* int (FILE *, const fpos_t *) */
  return reinterpret_cast<intptr_t>(std::addressof(fsetpos));
}


extern "C" intptr_t get_fseeko64_address() {
  /* int (FILE *, __off64_t, int) */
  return reinterpret_cast<intptr_t>(std::addressof(fseeko64));
}


extern "C" intptr_t get_ftello64_address() {
  /* __off64_t (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(ftello64));
}


extern "C" intptr_t get_fgetpos64_address() {
  /* int (FILE *__restrict, fpos64_t *__restrict) */
  return reinterpret_cast<intptr_t>(std::addressof(fgetpos64));
}


extern "C" intptr_t get_fsetpos64_address() {
  /* int (FILE *, const fpos64_t *) */
  return reinterpret_cast<intptr_t>(std::addressof(fsetpos64));
}


extern "C" intptr_t get_clearerr_address() {
  /* void (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clearerr));
}


extern "C" intptr_t get_feof_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(feof));
}


extern "C" intptr_t get_ferror_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ferror));
}


extern "C" intptr_t get_clearerr_unlocked_address() {
  /* void (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(clearerr_unlocked));
}


extern "C" intptr_t get_feof_unlocked_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(feof_unlocked));
}


extern "C" intptr_t get_ferror_unlocked_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ferror_unlocked));
}


extern "C" intptr_t get_perror_address() {
  /* void (const char *) */
  return reinterpret_cast<intptr_t>(std::addressof(perror));
}


extern "C" intptr_t get_fileno_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fileno));
}


extern "C" intptr_t get_fileno_unlocked_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(fileno_unlocked));
}


extern "C" intptr_t get_pclose_address() {
  /* int (FILE *) */
  return reinterpret_cast<intptr_t>(std::addressof(pclose));
}


extern "C" intptr_t get_popen_address() {
  /* FILE *(const char *, const char *) */
  return reinterpret_cast<intptr_t>(std::addressof(popen));
}


extern "C" intptr_t get_ctermid_address() {
  /* char *(char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ctermid));
}


extern "C" intptr_t get_cuserid_address() {
  /* char *(char *) */
  return reinterpret_cast<intptr_t>(std::addressof(cuserid));
}


extern "C" intptr_t get_obstack_printf_address() {
  /* int (struct obstack *__restrict, const char *__restrict, ...) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(obstack_printf));
}


extern "C" intptr_t get_obstack_vprintf_address() {
  /* int (struct obstack *__restrict, const char *__restrict, __va_list_tag *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(obstack_vprintf));
}


extern "C" intptr_t get_flockfile_address() {
  /* void (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(flockfile));
}


extern "C" intptr_t get_ftrylockfile_address() {
  /* int (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(ftrylockfile));
}


extern "C" intptr_t get_funlockfile_address() {
  /* void (FILE *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(funlockfile));
}


extern "C" intptr_t get_iswalnum_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswalnum));
}


extern "C" intptr_t get_iswalpha_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswalpha));
}


extern "C" intptr_t get_iswcntrl_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswcntrl));
}


extern "C" intptr_t get_iswdigit_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswdigit));
}


extern "C" intptr_t get_iswgraph_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswgraph));
}


extern "C" intptr_t get_iswlower_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswlower));
}


extern "C" intptr_t get_iswprint_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswprint));
}


extern "C" intptr_t get_iswpunct_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswpunct));
}


extern "C" intptr_t get_iswspace_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswspace));
}


extern "C" intptr_t get_iswupper_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswupper));
}


extern "C" intptr_t get_iswxdigit_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswxdigit));
}


extern "C" intptr_t get_iswblank_address() {
  /* int (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswblank));
}


extern "C" intptr_t get_wctype_address() {
  /* wctype_t (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctype));
}


extern "C" intptr_t get_iswctype_address() {
  /* int (wint_t, wctype_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswctype));
}


extern "C" intptr_t get_towlower_address() {
  /* wint_t (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towlower));
}


extern "C" intptr_t get_towupper_address() {
  /* wint_t (wint_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towupper));
}


extern "C" intptr_t get_wctrans_address() {
  /* wctrans_t (const char *) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctrans));
}


extern "C" intptr_t get_towctrans_address() {
  /* wint_t (wint_t, wctrans_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towctrans));
}


extern "C" intptr_t get_iswalnum_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswalnum_l));
}


extern "C" intptr_t get_iswalpha_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswalpha_l));
}


extern "C" intptr_t get_iswcntrl_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswcntrl_l));
}


extern "C" intptr_t get_iswdigit_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswdigit_l));
}


extern "C" intptr_t get_iswgraph_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswgraph_l));
}


extern "C" intptr_t get_iswlower_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswlower_l));
}


extern "C" intptr_t get_iswprint_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswprint_l));
}


extern "C" intptr_t get_iswpunct_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswpunct_l));
}


extern "C" intptr_t get_iswspace_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswspace_l));
}


extern "C" intptr_t get_iswupper_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswupper_l));
}


extern "C" intptr_t get_iswxdigit_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswxdigit_l));
}


extern "C" intptr_t get_iswblank_l_address() {
  /* int (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswblank_l));
}


extern "C" intptr_t get_wctype_l_address() {
  /* wctype_t (const char *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctype_l));
}


extern "C" intptr_t get_iswctype_l_address() {
  /* int (wint_t, wctype_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(iswctype_l));
}


extern "C" intptr_t get_towlower_l_address() {
  /* wint_t (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towlower_l));
}


extern "C" intptr_t get_towupper_l_address() {
  /* wint_t (wint_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towupper_l));
}


extern "C" intptr_t get_wctrans_l_address() {
  /* wctrans_t (const char *, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(wctrans_l));
}


extern "C" intptr_t get_towctrans_l_address() {
  /* wint_t (wint_t, wctrans_t, locale_t) noexcept(true) */
  return reinterpret_cast<intptr_t>(std::addressof(towctrans_l));
}
