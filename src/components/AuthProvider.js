'use client'
import { createContext, useContext, useEffect, useState } from 'react'
import { supabase } from '../lib/supabaseClient'

const AuthContext = createContext()

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Fetch initial session — handle session_not_found gracefully
    supabase.auth.getSession().then(({ data: { session }, error }) => {
      if (error) {
        console.warn('Session expired or invalid, clearing local auth:', error.message)
        // Clear any stale local tokens
        supabase.auth.signOut({ scope: 'local' })
        setUser(null)
      } else {
        setUser(session?.user ?? null)
      }
      setIsLoading(false)
    })

    const { data: authListener } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        if (event === 'TOKEN_REFRESHED' && !session) {
          // Token refresh failed (session revoked from another browser)
          setUser(null)
        } else if (event === 'SIGNED_OUT') {
          setUser(null)
        } else {
          setUser(session?.user ?? null)
        }
        setIsLoading(false)
      }
    )

    return () => {
      authListener.subscription.unsubscribe()
    }
  }, [])

  async function signInWithGoogle() {
    const { error } = await supabase.auth.signInWithOAuth({
      provider: 'google',
    })
    if (error) console.error('Error signing in with Google:', error)
  }

  async function signOut() {
    // Use scope: 'local' to avoid revoking the session for other browsers
    const { error } = await supabase.auth.signOut({ scope: 'local' })
    if (error) console.error('Error signing out:', error)
    setUser(null)
  }

  return (
    <AuthContext.Provider value={{ user, isLoading, signInWithGoogle, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  return useContext(AuthContext)
}
