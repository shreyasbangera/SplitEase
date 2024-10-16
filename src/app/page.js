'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../components/AuthProvider'
import { Button } from "../components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "../components/ui/card"
import { supabase } from '@/lib/supabaseClient'
import { useRequireAuth } from '@/components/useRequireAuth'

export default function Home() {
  const { user } = useRequireAuth()
  const router = useRouter()
  const [groups, setGroups] = useState([])

  console.log(user)

  useEffect(() => {
    if (user) fetchGroups()
  }, [user])

  async function fetchGroups() {
    const { data, error } = await supabase
    .from('groups')  
    .select('*')
    .or(`created_by.eq.${user.id},members.cs.{${user.email}}`);

  if (error) {
    console.error('Error fetching groups:', error)
  } else {
    setGroups(data)  // Assuming `setGroups` is a state setter to store the fetched data
  }
  }

  async function handleAddExpense() {
    if (groups.length > 0) {
      router.push('/add_expense')
    } else {
      alert('Please create a group first before adding an expense.')
    }
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Splitwise Clone</h1>
      
      <div className="mb-4">
        <Button onClick={() => router.push('/add_group')} className="mr-2">Add Group</Button>
        <Button onClick={handleAddExpense}>Add Expense</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Your Groups</CardTitle>
        </CardHeader>
        <CardContent>
          {groups.map((group) => (
            <div key={group.id} className="mb-2">
              <Button variant="link" onClick={() => router.push(`/group/${group.id}`)}>
                {group.name}
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}