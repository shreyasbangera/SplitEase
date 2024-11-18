'use client'
import React from "react";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuSeparator } from "./ui/dropdown-menu";
import { DropdownMenuTrigger } from "@radix-ui/react-dropdown-menu";
import Image from "next/image";
import { supabase } from "@/lib/supabaseClient";
import { useRouter } from "next/navigation";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "./AuthProvider";
import { ModeToggle } from "./ModeToggle";

const Navbar = () => {
    const  {user}  = useAuth()
    const router = useRouter()
    const { toast } = useToast()

    const handleSignOut = async () => {
        const { error } = await supabase.auth.signOut();
        if (error) {
          toast({ variant:"destructive", description: error.message });
        } else {
          router.push('/signin');
          toast({ title: "Signed Out", description :'You have been signed out successfully!' });
        }
      };

  return (
    <header className="fixed bg-white dark:bg-background w-full flex items-center justify-between border-b border-solid border-b-[#e7eef4] dark:border-b-gray-700 lg:px-10 px-4 py-3 h-[54px]">
      <div onClick={() => router.push('/')} className="flex items-center gap-4 cursor-pointer">
        <svg
          className="w-4 h-4"
          viewBox="0 0 48 48"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M42.4379 44C42.4379 44 36.0744 33.9038 41.1692 24C46.8624 12.9336 42.2078 4 42.2078 4L7.01134 4C7.01134 4 11.6577 12.932 5.96912 23.9969C0.876273 33.9029 7.27094 44 7.27094 44L42.4379 44Z"
            fill="currentColor"
          ></path>
        </svg>
        <h2 className="text-lg font-bold">SplitEase</h2>
      </div>
      <div className="flex items-center gap-4">
      <ModeToggle />
      {user && <DropdownMenu>
        <DropdownMenuTrigger className="focus-visible:outline-none flex items-center gap-3">
            <Image className="rounded-full aspect-square" src={user?.user_metadata.avatar_url} width={30} height={30} alt='avatar'/>
          <span className="text-sm lg:flex hidden">{user?.user_metadata.full_name}</span>
        </DropdownMenuTrigger>
        <DropdownMenuContent className='mr-5'>
          <DropdownMenuLabel className='font-medium'>{user?.user_metadata.full_name}</DropdownMenuLabel>
          <DropdownMenuLabel className='font-thin text-xs py-1'>{user?.user_metadata.email}</DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem onClick={handleSignOut} className='text-red-500 cursor-pointer'>Sign out</DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>}
      </div>
    </header>
  );
};

export default Navbar;
