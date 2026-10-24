export const motionEase = [0.22, 1, 0.36, 1] as const;

export const staggerItem = {
  hidden: { opacity: 0, y: 9 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.38, ease: motionEase },
  },
};
