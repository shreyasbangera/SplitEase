'use client'
import { useState } from 'react'
import { Button } from "../../components/ui/button"
import { Input } from "../../components/ui/input"
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "../../components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../components/ui/tabs"
import { supabase } from '../../lib/supabaseClient'
import { useRouter, useSearchParams } from 'next/navigation'
import { useRequireAuth } from '@/components/useRequireAuth'
import { useAuth } from '@/components/AuthProvider'

const page = () => {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const router = useRouter()
    const searchParams = useSearchParams()
    const returnUrl = searchParams.get('returnUrl') || '/'
    const user = useAuth()

    console.log(user)
  
    const handleSignUp = async (e) => {
      e.preventDefault()
      const { error } = await supabase.auth.signUp({ email, password })
      if (error) {
        alert(error.message)
      } else {
        alert('Check your email for the login link!')
      }
    }
  
    const handleSignIn = async (e) => {
      e.preventDefault()
      const { error } = await supabase.auth.signInWithPassword({ email, password })
      if (error) {
        alert(error.message)
      } else {
        console.log(returnUrl)
        router.push(returnUrl)
      }
    }
  
    const handleGoogleSignIn = async () => {
      const { error } = await supabase.auth.signInWithOAuth({
        provider: 'google',
        options: {
        redirectTo: `${window.location.origin}/${returnUrl}`
        }
      })
      if (error) {
        alert(error.message)
      }
    }

    console.log(returnUrl)
  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <Card className="w-[350px]">
        <CardHeader>
          <CardTitle>Welcome to Splitwise</CardTitle>
          <CardDescription>Login or create an account to get started</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="login">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="login">Login</TabsTrigger>
              <TabsTrigger value="signup">Signup</TabsTrigger>
            </TabsList>
            <TabsContent value="login">
              <form onSubmit={handleSignIn}>
                <div className="space-y-4">
                  <Input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <Input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <Button type="submit" className="w-full">Login</Button>
                </div>
              </form>
            </TabsContent>
            <TabsContent value="signup">
              <form onSubmit={handleSignUp}>
                <div className="space-y-4">
                  <Input
                    type="email"
                    placeholder="Email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                  <Input
                    type="password"
                    placeholder="Password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <Button type="submit" className="w-full">Sign Up</Button>
                </div>
              </form>
            </TabsContent>
          </Tabs>
        </CardContent>
        <CardFooter>
          <Button variant="outline" className="w-full" onClick={handleGoogleSignIn}>
            Sign in with Google
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}

export default page