'use client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../../components/AuthProvider'
import { supabase } from '../../../lib/supabaseClient'
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { useRequireAuth } from '@/components/useRequireAuth'

export default function JoinGroup({ params }) {
  const user = useRequireAuth()
  const router = useRouter()
  const [group, setGroup] = useState(null)

  useEffect(() => {
    if (user) {
      fetchGroup()
    }
  }, [user])

  async function fetchGroup() {
    const { data, error } = await supabase
      .from('groups')
      .select('*')
      .eq('id', params.id)
      .single()

    if (error) {
      console.error('Error fetching group:', error)
      return
    }

    setGroup(data)
  }

  async function handleJoinGroup() {
    if (!user || !group) return

    // Check if user is invited
    if (!group.invite_emails.includes(user.email)) {
      console.error('You are not invited to this group')
      return
    }

    const { data, error } = await supabase
      .from('groups')
      .update({ 
        members: [...group.members, user.id],
      })
      .eq('id', group.id)

    if (error) {
      console.error('Error joining group:', error)
      return
    }

    router.push(`/group/${group.id}`)
  }

  if (!group) return <div>Loading...</div>

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Join Group</h1>
      
      <Card>
        <CardHeader>
          <CardTitle>{group.name}</CardTitle>
        </CardHeader>
        <CardContent>
          <p>You've been invited to join this group.</p>
          <Button onClick={handleJoinGroup} className="mt-4">Join Group</Button>
        </CardContent>
      </Card>
    </div>
  )
}