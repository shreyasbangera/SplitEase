'use client'
import { Button } from "../../components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../../components/ui/card";
import { supabase } from '../../lib/supabaseClient';
import { useRouter, useSearchParams } from 'next/navigation';
import { Suspense } from 'react'; 
import googleIcon from '@/assets/google.png'
import Image from "next/image";
import { Skeleton } from "@/components/ui/skeleton";

const Loading = () => <Skeleton className="w-[100px] h-[20px] rounded-full" /> 

const Page = () => {
  const searchParams = useSearchParams();
  const returnUrl = searchParams.get('returnUrl') || '/';

  const handleGoogleSignIn = async () => {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/${returnUrl}`
      }
    });
    if (error) {
      alert(error.message);
    }
  };

  return (
    <div className="flex items-center justify-center bg-gray-100 h-[calc(100vh-54px)]">
      <Card className="lg:w-[400px] w-[90%]">
        <CardHeader>
          <CardTitle>Welcome to SplitEase</CardTitle>
          <CardDescription>Sign in or create an account to get started</CardDescription>
        </CardHeader>
        <CardFooter>
          <Button variant="outline" className="w-full flex gap-2" onClick={handleGoogleSignIn}>
          <Image src={googleIcon} alt='google' width={16} height={16} />
            Sign in with Google
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};

const SuspenseWrapper = () => (
  <Suspense fallback={<Loading />}>
    <Page />
  </Suspense>
);

export default SuspenseWrapper;