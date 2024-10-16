"use client"
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from "../../components/ui/button"
import { Input } from "../../components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import { Card, CardHeader, CardTitle, CardContent } from "../../components/ui/card"
import { supabase } from '../../lib/supabaseClient'
import { useAuth } from '@/components/AuthProvider'

export default function AddExpensePage() {
  const [description, setDescription] = useState('')
  const [amount, setAmount] = useState('')
  const [groupId, setGroupId] = useState('')
  const [groups, setGroups] = useState([])
  const router = useRouter()
  const { user } = useAuth()

  useEffect(() => {
    fetchGroups()
  }, [])

  async function fetchGroups() {
    const { data, error } = await supabase
      .from('groups')
      .select('id, name')
    
    if (error) {
      console.error('Error fetching groups:', error)
    } else if (data) {
      setGroups(data)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const { data: { user } } = await supabase.auth.getUser()
    if (!user) {
      alert('You must be logged in to add an expense')
      return
    }
    
    const { data, error } = await supabase
      .from('expenses')
      .insert({
        description,
        amount: parseFloat(amount),
        group_id: groupId,
        paid_by: user.id,
        created_at: new Date().toISOString()
      })
    
    if (error) {
      alert('Error adding expense: ' + error.message)
    } else {
      router.push('/')
    }
  }

  return (
    <div className="container mx-auto p-4">
      <Card className="max-w-md mx-auto">
        <CardHeader>
          <CardTitle>Add New Expense</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Input
              placeholder="Description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
            <Input
              type="number"
              placeholder="Amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
            <Select value={groupId} onValueChange={setGroupId}>
              <SelectTrigger>
              <SelectValue placeholder="Select a group" />
              </SelectTrigger>
              <SelectContent>
                {groups.map(({id, name}) => (
                  <SelectItem key={id} value={String(id)}>{name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button type="submit" className="w-full">Add Expense</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}