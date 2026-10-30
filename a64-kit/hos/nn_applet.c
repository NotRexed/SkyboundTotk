/* nn::oe (applet), nn::account, nn::time, nn::pl, nn::err: what a game asks the system about itself.
   The answers describe a docked console in English (HOS_LANGUAGE / language.txt: another language) with one user,
   "Player". */
#include "hos.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>   /* (the exe's folder; the browser build: the Microsoft C runtime's time functions) */

static int64_t boot_ns;

/* ---- nn::oe ------------------------------------------------------------------------------------------------- */
HLE(_ZN2nn2oe10InitializeEv) { if (!boot_ns) boot_ns = hos_now_ns(); }
HLE(_ZN2nn2oe17FinishStartupLogoEv) { hos_log("oe: startup logo finished"); }
/* GetDesiredLanguage() -> settings::LanguageCode (char[8]) by value: the language the game shows. English (US)
   unless HOS_LANGUAGE or the first line of language.txt beside the exe names one TOTK has (after SkyboundTotk PR 16
   by sardigital): ja, en-US, en-GB, fr, fr-CA, de, it, es, es-419, nl, ru, ko, zh-CN / zh-Hans, zh-TW / zh-Hant. */
static const char *desired_language(void) {
    static char lang[8];
    if (lang[0]) return lang;
    static const char *const known[] = {"ja", "en-US", "en-GB", "fr",    "fr-CA", "de",      "it",     "es",
                                        "es-419", "nl", "ru",   "ko",    "zh-CN", "zh-Hans", "zh-TW", "zh-Hant"};
    char want[32] = "";
    const char *from = "the default";
    if (getenv("HOS_LANGUAGE")) {
        snprintf(want, sizeof want, "%s", getenv("HOS_LANGUAGE"));
        from = "HOS_LANGUAGE";
    } else {
        char path[300] = "language.txt";
#if defined(_WIN32) && !defined(__EMSCRIPTEN__)
        DWORD n = GetModuleFileNameA(NULL, path, 260);
        while (n > 0 && path[n - 1] != '\\' && path[n - 1] != '/') n--;
        snprintf(path + n, sizeof path - n, "language.txt");
#endif
        FILE *f = fopen(path, "r");
        if (f) {
            if (fgets(want, sizeof want, f)) from = "language.txt";
            fclose(f);
        }
    }
    want[strcspn(want, " \t\r\n")] = 0;
    snprintf(lang, sizeof lang, "en-US");
    for (size_t i = 0; want[0] && i < sizeof known / sizeof *known; i++)
        if (!strcmp(want, known[i])) snprintf(lang, sizeof lang, "%s", known[i]);
    if (want[0] && strcmp(want, lang)) hos_log("oe: language \"%s\" (%s) is not one TOTK has: English (US)", want, from);
    else hos_log("oe: language %s (%s)", lang, from);
    return lang;
}
HLE(_ZN2nn2oe18GetDesiredLanguageEv) {
    uint64_t v = 0;
    memcpy(&v, desired_language(), strlen(desired_language()));
    RET(v);
}
HLE(_ZN2nn2oe16GetOperationModeEv) { RET(1); }        /* OperationMode_Console (docked) */
HLE(_ZN2nn2oe18GetPerformanceModeEv) { RET(1); }      /* PerformanceMode_Boost (docked) */
HLE(_ZN2nn2oe20GetCurrentFocusStateEv) { RET(1); }    /* FocusState_InFocus */
HLE(_ZN2nn2oe25GetProgramTotalActiveTimeEv) { RET(hos_now_ns() - boot_ns); }
HLE(_ZN2nn2oe24GetExpectedVolumeBalanceEPfS1_) {
    float one = 1.0f;
    memcpy(GP(ARG(0)), &one, 4);
    memcpy(GP(ARG(1)), &one, 4);
}
HLE(_ZN2nn2oe24SetExpectedVolumeBalanceEff) {}
HLE(_ZN2nn2oe15SetCpuBoostModeENS0_12CpuBoostModeE) {}
HLE(_ZN2nn2oe20SetFocusHandlingModeENS0_17FocusHandlingModeE) {}
HLE(_ZN2nn2oe27SetPerformanceConfigurationENS0_15PerformanceModeEi) {}
HLE(_ZN2nn2oe42SetOperationModeChangedNotificationEnabledEb) {}
HLE(_ZN2nn2oe44SetPerformanceModeChangedNotificationEnabledEb) {}
HLE(_ZN2nn2oe28SetResumeNotificationEnabledEb) {}
HLE(_ZN2nn2oe38SetUserInactivityDetectionTimeExtendedEb) {}
/* No system notifications ever arrive: TryPop says none, Pop waits forever (as it would on a quiet console). */
HLE(_ZN2nn2oe25TryPopNotificationMessageEPj) { RET(0); }
HLE(_ZN2nn2oe22PopNotificationMessageEv) { for (;;) hos_sleep_ns(1000000000LL); }

/* ---- nn::account: one user ------------------------------------------------------------------------------------ */
static const uint64_t user_uid[2] = {0x0000000000000001ULL, 0x5a3c9e1f0b7d2468ULL};
HLE(_ZN2nn7account10InitializeEv) {}
/* TryOpenPreselectedUser(UserHandle*) -> bool. A handle is three words; ours carries the uid. */
HLE(_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE) {
    G64(ARG(0)) = user_uid[0];
    G64(ARG(0) + 8) = user_uid[1];
    G64(ARG(0) + 16) = 1;
    RET(1);
}
HLE(_ZN2nn7account9CloseUserERKNS0_10UserHandleE) {}
/* GetUserId(Uid*, UserHandle const&) -> Result */
HLE(_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE) {
    G64(ARG(0)) = user_uid[0];
    G64(ARG(0) + 8) = user_uid[1];
    RET(0);
}
/* GetNickname(Nickname*, Uid const&) -> Result: Nickname is char[33] */
HLE(_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE) {
    memset(GP(ARG(0)), 0, 33);
    memcpy(GP(ARG(0)), "Player", 6);
    RET(0);
}
HLE(_ZN2nn7account16GetUserExistenceEPbRKNS0_3UidE) {
    *(uint8_t *)GP(ARG(0)) = G64(ARG(1)) == user_uid[0] && G64(ARG(1) + 8) == user_uid[1];
    RET(0);
}

/* ---- nn::time: the host clock --------------------------------------------------------------------------------- */
HLE(_ZN2nn4time10InitializeEv) { RET(0); }
HLE(_ZN2nn4time8FinalizeEv) { RET(0); }
HLE(_ZN2nn4time13IsInitializedEv) { RET(1); }
HLE(_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE) { G64(ARG(0)) = (uint64_t)time(NULL); RET(0); }
HLE(_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE) { G64(ARG(0)) = (uint64_t)time(NULL); RET(0); }
/* StandardSteadyClock::now() -> SteadyClockTimePoint { s64 seconds; Uuid source[16] }: 24 bytes, returned via x8 */
HLE(_ZN2nn4time19StandardSteadyClock3nowEv) {
    memset(GP(c->x[8]), 0, 24);
    G64(c->x[8]) = (uint64_t)(hos_now_ns() / 1000000000LL);
    G64(c->x[8] + 8) = 0x484f53434c4f434bULL;   /* a fixed source id: "HOSCLOCK" */
}

static void calendar(uint64_t out, uint64_t info, int64_t posix, int local) {
    time_t t = (time_t)posix;
    struct tm tmv;
    if (local) localtime_s(&tmv, &t); else gmtime_s(&tmv, &t);
    /* CalendarTime { s16 year; s8 month, day, hour, minute, second; pad } */
    uint8_t ct[8] = {0};
    int16_t year = (int16_t)(tmv.tm_year + 1900);
    memcpy(ct, &year, 2);
    ct[2] = (uint8_t)(tmv.tm_mon + 1); ct[3] = (uint8_t)tmv.tm_mday; ct[4] = (uint8_t)tmv.tm_hour;
    ct[5] = (uint8_t)tmv.tm_min; ct[6] = (uint8_t)tmv.tm_sec;
    memcpy(GP(out), ct, 8);
    if (info) {   /* CalendarAdditionalInfo { s32 dayOfWeek, dayOfYear; char tz[8]; s32 isDst; s32 utcOffset } */
        memset(GP(info), 0, 0x18);
        G32(info) = (uint32_t)tmv.tm_wday;
        G32(info + 4) = (uint32_t)tmv.tm_yday;
        memcpy(GP(info + 8), "UTC", 3);
    }
}
/* ToCalendarTime(CalendarTime*, CalendarAdditionalInfo*, PosixTime const&) -> Result */
HLE(_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE) {
    calendar(ARG(0), ARG(1), (int64_t)G64(ARG(2)), 1);
    RET(0);
}
/* ToCalendarTimeInUtc(PosixTime const&) -> CalendarTime (8 bytes, in x0) */
HLE(_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE) {
    static _Thread_local uint64_t tmp;
    if (!tmp) tmp = hos_sys_alloc(8, 8);
    calendar(tmp, 0, (int64_t)G64(ARG(0)), 0);
    RET(G64(tmp));
}
/* ToPosixTimeFromUtc(CalendarTime const&) -> PosixTime */
HLE(_ZN2nn4time18ToPosixTimeFromUtcERKNS0_12CalendarTimeE) {
    uint8_t ct[8];
    memcpy(ct, GP(ARG(0)), 8);
    struct tm tmv = {0};
    int16_t year;
    memcpy(&year, ct, 2);
    tmv.tm_year = year - 1900; tmv.tm_mon = ct[2] - 1; tmv.tm_mday = ct[3];
    tmv.tm_hour = ct[4]; tmv.tm_min = ct[5]; tmv.tm_sec = ct[6];
    RET((uint64_t)_mkgmtime(&tmv));
}
/* ClockSnapshot: an opaque 0xD0-byte record; ours holds the POSIX time at +0 */
HLE(_ZN2nn4time13ClockSnapshotC1Ev) { memset(GP(ARG(0)), 0, 0xD0); }
HLE(_ZN2nn4time13ClockSnapshot29CreateWithStandardSystemClockEPS1_) {
    memset(GP(ARG(0)), 0, 0xD0);
    G64(ARG(0)) = (uint64_t)time(NULL);
    RET(0);
}
/* CalculateSpanBetween(TimeSpan*, ClockSnapshot const& a, ClockSnapshot const& b) -> Result */
HLE(_ZN2nn4time20CalculateSpanBetweenEPNS_8TimeSpanERKNS0_13ClockSnapshotES5_) {
    G64(ARG(0)) = (uint64_t)(((int64_t)G64(ARG(2)) - (int64_t)G64(ARG(1))) * 1000000000LL);
    RET(0);
}
HLE(_ZN2nn4time48CalculateStandardUserSystemClockDifferenceByUserERKNS0_13ClockSnapshotES3_) { RET(0); }

/* ---- nn::pl: system shared fonts. We have none; report an empty one, loaded. ------------------------------------ */
static uint64_t empty_font;
HLE(_ZN2nn2pl22GetSharedFontLoadStateENS0_14SharedFontTypeE) { RET(1); }   /* SharedFontLoadState_Loaded */
HLE(_ZN2nn2pl20GetSharedFontAddressENS0_14SharedFontTypeE) {
    if (!empty_font) empty_font = hos_sys_alloc(0x1000, 0x1000);
    RET(empty_font);
}
HLE(_ZN2nn2pl17GetSharedFontSizeENS0_14SharedFontTypeE) { RET(0); }

/* ---- nn::err ------------------------------------------------------------------------------------------------- */
HLE(_ZN2nn3err9ShowErrorENS_6ResultE) { hos_log("err: the game shows error %#x (module %u, description %u)", (uint32_t)ARG(0),
                                                (uint32_t)ARG(0) & 0x1FF, ((uint32_t)ARG(0) >> 9) & 0x1FFF); }

/* ---- nn::prepo: play reports. The PlayReport builder (Add*, SetEventId) is the SDK's own code; saving one would
   send it to the prepo service. Reports are dropped. ---------------------------------------------------------- */
HLE(_ZN2nn5prepo10PlayReport4SaveEv) { RET(0); }
HLE(_ZN2nn5prepo10PlayReport4SaveERKNS_7account3UidE) { RET(0); }
