import { MessageSquarePlus, History, Settings, X, ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { motion, AnimatePresence } from 'motion/react';

interface AmanaSidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
}

export function AmanaSidebar({ isOpen, onClose, onNewChat }: AmanaSidebarProps) {
  const sidebarContent = (
    <div className="flex flex-col h-full w-full bg-slate-50/50 border-r py-4 px-3 gap-2 dark:bg-background/80">
      <div className="flex items-center justify-between mb-2 md:hidden">
        <span className="font-semibold text-sm px-2 text-muted-foreground">Menu</span>
        <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <Button onClick={onNewChat} className="justify-start gap-2 h-10 font-semibold" variant="default">
        <MessageSquarePlus className="w-4 h-4" />
        Chat Baru
      </Button>

      <div className="mt-4 flex-1 overflow-y-auto">
        <h3 className="px-2 text-xs font-semibold text-muted-foreground mb-2 uppercase tracking-wider">Riwayat</h3>
        <div className="text-sm text-muted-foreground px-2 py-3 flex items-center gap-2">
            <History className="h-4 w-4" />
            <span className="italic">Belum ada riwayat chat</span>
        </div>
      </div>

      <div className="mt-auto px-2 py-4 border-t">
         <div className="flex items-center gap-3 text-sm text-muted-foreground">
             <ShieldAlert className="h-4 w-4" />
             <span className="text-xs leading-tight">Jawaban dihasilkan oleh AI, harap verifikasi kesesuaian fatwa.</span>
         </div>
      </div>
      <Button variant="ghost" className="justify-start gap-2 h-10 text-muted-foreground mt-2">
        <Settings className="w-4 h-4" />
        Pengaturan
      </Button>
    </div>
  );

  return (
    <>
      {/* Mobile Backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/20 z-40 md:hidden backdrop-blur-sm"
          />
        )}
      </AnimatePresence>

      {/* Mobile Sidebar */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', bounce: 0, duration: 0.3 }}
            className="fixed inset-y-0 left-0 w-72 z-50 md:hidden bg-background shadow-xl"
          >
            {sidebarContent}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Desktop Sidebar */}
      <div className="hidden md:block w-[260px] lg:w-[280px] shrink-0 h-full">
        {sidebarContent}
      </div>
    </>
  );
}
