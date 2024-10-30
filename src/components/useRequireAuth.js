import { useEffect } from "react";
import { useAuth } from "./AuthProvider"; // Assumes you have a custom AuthProvider hook
import { usePathname, useRouter } from "next/navigation";

export function useRequireAuth() {
  const { user, isLoading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  
  useEffect(() => {
    if (!isLoading) {
      if (!user) {
        router.push(`/login?returnUrl=${encodeURIComponent(pathname)}`);
      }
    }
  }, [user, isLoading, pathname, router]);

  return user;
}
