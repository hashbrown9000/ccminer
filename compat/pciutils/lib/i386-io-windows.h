/*
 *	The PCI Library -- Access to i386 I/O ports on Windows
 *
 *	Copyright (c) 2004 Alexander Stock <stock.alexander@gmx.de>
 *	Copyright (c) 2006 Martin Mares <mj@ucw.cz>
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 *
 *	Changes:
 *	2015-06-23: Proper implementation (tpruvot)
 *
 */

#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>

/* WinIo declarations */
typedef BOOL bool;

typedef bool (_stdcall *INITIALIZEWINIO)(void);
typedef void (_stdcall *SHUTDOWNWINIO)(void);
typedef bool (_stdcall *GETPORTVAL)(WORD, PDWORD, BYTE);
typedef bool (_stdcall *SETPORTVAL)(WORD, DWORD, BYTE);

INITIALIZEWINIO InitializeWinIo;
SHUTDOWNWINIO ShutdownWinIo;
GETPORTVAL GetPortVal;
SETPORTVAL SetPortVal;

#ifdef _WIN64
#define WINIO_LIB_NAME_V3 "WinIo64.dll"
#else
#define WINIO_LIB_NAME_V3 "WinIo32.dll"
#endif

#define GETPROC(n,d) \
  n = (d) GetProcAddress(lib, #n); \
  if (!n) { \
    a->warning("i386-io-windows: Couldn't find " #n " function."); \
    return 0; \
  }

static HMODULE lib = NULL;

static int
intel_setup_io(struct pci_access *a)
{
  // Check if already loaded
  if (!lib)
  {
    lib = LoadLibrary(WINIO_LIB_NAME_V3);
    if (!lib) /* WinIo 3 loading failed, try loading WinIo 2 */
      lib = LoadLibrary("WinIo.dll");

    if (!lib) {
      a->warning("i386-io-windows: Neither " WINIO_LIB_NAME_V3 " or WinIo.dll could be loaded.");
      return 0;
    }

    GETPROC(InitializeWinIo, INITIALIZEWINIO);
    GETPROC(ShutdownWinIo, SHUTDOWNWINIO);
    GETPROC(GetPortVal, GETPORTVAL);
    GETPROC(SetPortVal, SETPORTVAL);
  }

  if (!InitializeWinIo()) {
    a->warning("i386-io-windows: IO library initialization failed. Try running from an elevated command prompt.");
    return 0;
  }

  return 1;
}

static inline int
intel_cleanup_io(struct pci_access *a UNUSED)
{
  if (!lib)
    return 0;
  ShutdownWinIo();
  FreeLibrary(lib);
  lib = NULL;
  return -1;
}

static inline u8
inb (u16 port)
{
  if (lib) {
    DWORD pv;
    if (GetPortVal(port, &pv, 1))
      return (u8)pv;
  }
  return 0;
}

static inline u16
inw (u16 port)
{
  if (lib) {
    DWORD pv;
    if (GetPortVal(port, &pv, 2))
      return (u16)pv;
  }
  return 0;
}

static inline u32
inl (u16 port)
{
  if (lib) {
    DWORD pv;
    if (GetPortVal(port, &pv, 4))
      return (u32)pv;
  }
  return 0;
}

static inline void
outb (u8 value, u16 port)
{
  if (lib) {
    SetPortVal(port, value, 1);
  }
}

static inline void
outw (u16 value, u16 port)
{
  if (lib) {
    SetPortVal(port, value, 2);
  }
}

static inline void
outl (u32 value, u16 port)
{
  if (lib) {
    SetPortVal(port, value, 4);
  }
}
