#define PCI_CONFIG_H
#define PCI_ARCH_I386

#define PCI_OS_WINDOWS

#define PCI_HAVE_PM_DUMP
#define PCI_HAVE_PM_INTEL_CONF

#define PCI_IDS "pci.ids"
#define PCI_PATH_IDS_DIR "."
#define PCI_PATH_PROC_BUS_PCI ""
#define PCI_PATH_SYS_BUS_PCI ""
#define PCILIB_VERSION "3.3.1"

// #ifndef _DEBUG
// #undef PCI_PRINTF /*(x,y)*/
// #endif

#ifdef _MSC_VER
//#define PCI_HAVE_STDINT_H
#define R_OK 4 /* access() read mode */
#define lseek _lseek
#endif
