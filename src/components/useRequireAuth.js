import { useEffect } from "react";
import { useAuth } from "./AuthProvider";
import { usePathname, useRouter } from "next/navigation";

export function useRequireAuth(){
    const {user, isLoading} = useAuth();
    const router = useRouter()
    const pathname = usePathname()

    console.log('use', isLoading)

    useEffect(() => {
        const timer = setTimeout(() => {
              if (!user) {
                setIsAuthorized(false);
                router.push(`/login?returnUrl=${encodeURIComponent(pathname)}`);
              }
          }, 5000); // 5 seconds timeout
      
          return () => clearTimeout(timer);
    },[user])

    return user;
}