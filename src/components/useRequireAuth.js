import { useEffect } from "react";
import { useAuth } from "./AuthProvider";
import { usePathname, useRouter } from "next/navigation";

export function useRequireAuth(){
    const {user, isLoading} = useAuth();
    const router = useRouter()
    const pathname = usePathname()

    useEffect(() => {
        const timer = setTimeout(() => {
              if (!user) {
                router.push(`/login?returnUrl=${encodeURIComponent(pathname)}`);
              }
          }, 1000);
      
          return () => clearTimeout(timer);
    },[user])

    return user;
}