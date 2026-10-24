import type { ReactNode } from "react";
import { motion, useReducedMotion } from "motion/react";
import { motionEase } from "@/lib/motion";

export function Reveal({
  children,
  className = "",
  delay = 0,
  direction = "up",
  stagger = 0.07,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  direction?: "up" | "left" | "right";
  stagger?: number;
}) {
  const reduced = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduced ? false : "hidden"}
      whileInView="visible"
      viewport={{ once: true, amount: 0.12 }}
      variants={{
        hidden: {
          opacity: 0,
          x: direction === "left" ? -16 : direction === "right" ? 16 : 0,
          y: direction === "up" ? 16 : 0,
        },
        visible: {
          opacity: 1,
          x: 0,
          y: 0,
          transition: {
            duration: reduced ? 0 : 0.55,
            delay: reduced ? 0 : delay,
            staggerChildren: reduced ? 0 : stagger,
            ease: motionEase,
          },
        },
      }}
    >
      {children}
    </motion.div>
  );
}
