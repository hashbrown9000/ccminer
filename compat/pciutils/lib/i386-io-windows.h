/*
 *	The PCI Library -- Access to i386 I/O ports on Windows
 *
 *	Copyright (c) 2004 Alexander Stock <stock.alexander@gmx.de>
 *	Copyright (c) 2006 Martin Mares <mj@ucw.cz>
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 */

#include <io.h>
#include <windows.h>
#include <errno.h>

#ifndef __GNUC__
#include <conio.h>
#else
int _outp(unsigned short port, int databyte);
unsigned short _outpw(unsigned short port, unsigned short dataword);
unsigned long _outpd(unsigned short port, unsigned long dataword);
int _inp(unsigned short port);
unsigned short _inpw(unsigned short port);
unsigned long _inpd(unsigned short port);
#endif

#define outb(x,y) _outp(y,x)
#define outw(x,y) _outpw(y,x)
#define outl(x,y) _outpd(y,x)

#define inb(x) _inp(x)
#define inw(x) _inpw(x)
#define inl(x) _inpd(x)

#ifdef _WIN64
#define WINIO_LIB_NAME "WinIo64.dll"
#else
#define WINIO_LIB_NAME "WinIo32.dll"
#endif

static HMODULE lib = NULL;

static int
intel_setup_io(struct pci_access *a)
{
  typedef int (*MYPROC)(void);
  MYPROC InitializeWinIo;

  lib = LoadLibrary(WINIO_LIB_NAME);
  if (!lib)
    {
      lib = LoadLibrary("WinIo.dll");
    }
  if (!lib)
    {
      a->warning("i386-io-windows: Couldn't load " WINIO_LIB_NAME);
      return -ENOENT;
    }

  /* XXX: Is this really needed? --mj */
  GetProcAddress(lib, "InitializeWinIo");

  InitializeWinIo = (MYPROC) GetProcAddress(lib, "InitializeWinIo");
  if (!InitializeWinIo)
    {
      a->warning("i386-io-windows: Couldn't find InitializeWinIo function.");
      return -ENOSYS;
    }

  if (!InitializeWinIo())
    {
      a->warning("i386-io-windows: InitializeWinIo() failed.");
      return 0;
    }

  return 1;
}

static inline int
intel_cleanup_io(struct pci_access *a UNUSED)
{
  //TODO: ShutdownWinIO
  return 1;
}
