/*
 *	The PCI Library -- Access to i386 I/O ports on Windows
 *
 *	Copyright (c) 2011-2012 Jernej Simoncic <jernej+s-pciutils@eternallybored.org>
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 *
 * Changelog:
 * 2015-06-22: Define the WinIo (v3) dll name and make it works twice (tpruvot)
 * 2012-02-19: added back support for WinIo when DirectIO isn't available
 * 2011-06-20: original release for DirectIO library
 */

#define WIN32_LEAN_AND_MEAN
#define BUILD_WINDOWS /* deprecated GetVersion() */
#define NOGDI
#include <windows.h>

typedef enum {
	psByte  = 1,
	psWord  = 2,
	psDWord = 3
} DIRECTIO_PORT_SIZE;

typedef BOOL (WINAPI *SETDLLDIRECTORY)(LPCSTR);

#define LIB_DIRECTIO 1
#define LIB_WINIO 2
int lib_used;

/* DirectIO declarations */
typedef BOOL (WINAPI *DIRECTIO_INIT)(void);
typedef void (WINAPI *DIRECTIO_DEINIT)(void);
typedef BOOL (WINAPI *DIRECTIO_WRITEPORT)(ULONG,USHORT,DIRECTIO_PORT_SIZE);
typedef BOOL (WINAPI *DIRECTIO_READPORT)(PULONG,USHORT,DIRECTIO_PORT_SIZE);

DIRECTIO_INIT DirectIO_Init;
DIRECTIO_DEINIT DirectIO_DeInit;
DIRECTIO_WRITEPORT DirectIO_WritePort;
DIRECTIO_READPORT DirectIO_ReadPort;

/* WinIo declarations */
typedef BYTE bool;

typedef bool (_stdcall* INITIALIZEWINIO)(void);
typedef void (_stdcall* SHUTDOWNWINIO)(void);
typedef bool (_stdcall* GETPORTVAL)(WORD,PDWORD,BYTE);
typedef bool (_stdcall* SETPORTVAL)(WORD,DWORD,BYTE);

SHUTDOWNWINIO ShutdownWinIo;
GETPORTVAL GetPortVal;
SETPORTVAL SetPortVal;
INITIALIZEWINIO InitializeWinIo;


#ifdef _WIN64
#define WINIO_LIB_NAME_V3 "WinIo64.dll"
#else
#define WINIO_LIB_NAME_V3 "WinIo32.dll"
#endif

static HMODULE lib = NULL;

static int
intel_setup_io(struct pci_access *a)
{
  SETDLLDIRECTORY fnSetDllDirectory;

  /* remove current directory from DLL search path */
  fnSetDllDirectory = GetProcAddress(GetModuleHandle("kernel32"), "SetDllDirectoryA");
  if (fnSetDllDirectory)
    fnSetDllDirectory("");

#ifndef WIN64
  if ((GetVersion() & 0x80000000) == 0)
    { /* running on NT, try DirectIo first */
      lib = LoadLibrary("DirectIOLib32.dll");
#else
      lib = LoadLibrary("DirectIOLibx64.dll");
#endif
      if (!lib)
        { /* DirectIO loading failed, try WinIO instead */
          lib = LoadLibrary(WINIO_LIB_NAME_V3);
          if (!lib)
            { /* WinIo 3 loading failed, try loading WinIo 2 */
              lib = LoadLibrary("WinIo.dll");
            }
          if (!lib)
            {
              a->warning("i386-io-windows: Neither DirectIO, nor WinIo library could be loaded.\n");
              return 0;
            }
          else
              lib_used = LIB_WINIO;
        }
      else
          lib_used = LIB_DIRECTIO;
#ifndef WIN64
    }
  else
    { /* running on Win9x, only try loading WinIo 2 */
      lib = LoadLibrary("WinIo.dll");
      if (!lib)
        {
          a->warning("i386-io-windows: WinIo library could not be loaded.\n");
          return 0;
        }
      else
          lib_used = LIB_WINIO;
    }
#endif

#define GETPROC(n,d)   n = (d) GetProcAddress(lib, #n); \
                               if (!n) \
                                 { \
                                   a->warning("i386-io-windows: Couldn't find " #n " function.\n"); \
                                   return 0; \
                                 }


  if (lib_used == LIB_DIRECTIO)
    {
      GETPROC(DirectIO_Init, DIRECTIO_INIT);
      GETPROC(DirectIO_DeInit, DIRECTIO_DEINIT);
      GETPROC(DirectIO_WritePort, DIRECTIO_WRITEPORT);
      GETPROC(DirectIO_ReadPort, DIRECTIO_READPORT);
    }
  else
    {
      GETPROC(InitializeWinIo, INITIALIZEWINIO);
      GETPROC(ShutdownWinIo, SHUTDOWNWINIO);
      GETPROC(GetPortVal, GETPORTVAL);
      GETPROC(SetPortVal, SETPORTVAL);
    }

  if (!((lib_used == LIB_DIRECTIO) ? DirectIO_Init() : InitializeWinIo()))
    {
      a->warning("i386-io-windows: IO library initialization failed. Try running from an elevated command prompt.\n");
      return 0;
    }

  return 1;
}

static inline int
intel_cleanup_io(struct pci_access *a UNUSED)
{
  if (lib_used == LIB_DIRECTIO)
      DirectIO_DeInit();
  else
      ShutdownWinIo();
  return -1;
}

static inline u8
inb (u16 port)
{
  DWORD pv;

  if (lib_used == LIB_DIRECTIO)
    {
      if (DirectIO_ReadPort(&pv, port, psByte))
          return (u8)pv;
    }
  else
    {
      if (GetPortVal(port, &pv, 1))
          return (u8)pv;
    }
  return 0;
}

static inline u16
inw (u16 port)
{
  DWORD pv;

  if (lib_used == LIB_DIRECTIO)
    {
      if (DirectIO_ReadPort(&pv, port, psWord))
          return (u16)pv;
    }
  else
    {
      if (GetPortVal(port, &pv, 2))
          return (u16)pv;
    }
  return 0;
}

static inline u32
inl (u16 port)
{
  DWORD pv;

  if (lib_used == LIB_DIRECTIO)
    {
      if (DirectIO_ReadPort(&pv, port, psDWord))
          return (u32)pv;
    }
  else
    {
      if (GetPortVal(port, &pv, 4))
        return (u32)pv;
    }
  return 0;
}

static inline void
outb (u8 value, u16 port)
{
  if (lib_used == LIB_DIRECTIO)
      DirectIO_WritePort(value, port, psByte);
  else
      SetPortVal(port, value, 1);
}

static inline void
outw (u16 value, u16 port)
{
  if (lib_used == LIB_DIRECTIO)
      DirectIO_WritePort(value, port, psWord);
  else
      SetPortVal(port, value, 2);
}

static inline void
outl (u32 value, u16 port)
{
  if (lib_used == LIB_DIRECTIO)
      DirectIO_WritePort(value, port, psDWord);
  else
      SetPortVal(port, value, 4);
}
