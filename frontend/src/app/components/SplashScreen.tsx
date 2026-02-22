"use client";

import Image from "next/image";
import { useEffect, useState } from "react";

type SplashScreenProps = {
  children: React.ReactNode;
};

export default function SplashScreen({ children }: SplashScreenProps) {
  const [phase, setPhase] = useState<"logo" | "reveal" | "done">("logo");

  useEffect(() => {
    // Show logo for 1.6s, then start the reveal transition
    const t1 = setTimeout(() => setPhase("reveal"), 1600);
    // Remove splash from DOM after reveal animation completes
    const t2 = setTimeout(() => setPhase("done"), 2400);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, []);

  return (
    <>
      {phase !== "done" && (
        <div
          className="fixed inset-0 z-[9999] flex flex-col items-center justify-center bg-zinc-900 transition-all duration-700 ease-in-out"
          style={{
            opacity: phase === "reveal" ? 0 : 1,
            transform: phase === "reveal" ? "scale(1.08)" : "scale(1)",
            pointerEvents: phase === "reveal" ? "none" : "auto",
          }}
        >
          <div
            className="flex flex-col items-center gap-5 transition-all duration-700 ease-out"
            style={{
              opacity: phase === "logo" ? 1 : 0,
              transform: phase === "logo" ? "translateY(0) scale(1)" : "translateY(-20px) scale(1.1)",
            }}
          >
            <Image
              src="/logo.png"
              alt="Atlas"
              width={120}
              height={120}
              priority
              className="animate-[splashFadeIn_0.8s_ease-out_both]"
            />
            <h1 className="text-2xl font-semibold tracking-[0.15em] text-zinc-100 animate-[splashTextIn_0.6s_ease-out_0.4s_both]">
              ATLAS
            </h1>
            <div className="h-0.5 w-16 rounded-full bg-zinc-600 animate-[splashBarIn_0.5s_ease-out_0.7s_both]" />
          </div>
        </div>
      )}
      <div
        className="transition-opacity duration-500 ease-in"
        style={{ opacity: phase === "done" ? 1 : 0 }}
      >
        {children}
      </div>
    </>
  );
}
