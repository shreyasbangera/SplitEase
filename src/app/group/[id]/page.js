'use client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '../../../components/AuthProvider'
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { supabase } from '@/lib/supabaseClient'

export default function GroupDetails({ params }) {
  const { user } = useAuth()
  const router = useRouter()
  const [group, setGroup] = useState(null)
  const [expenses, setExpenses] = useState([])
  const [totalPending, setTotalPending] = useState(0)

  useEffect(() => {
      if (user && params.id) {
      fetchGroup()
      fetchExpenses()
    }
  }, [user, params.id])

  async function fetchGroup() {
    const { data, error } = await supabase
      .from('groups')
      .select('*')
      .eq('id', params.id)
      .single() // Assuming only one group needs to be fetched by `id`
  
    if (error) {
      console.error('Error fetching group:', error)
      return
    }
    setGroup(data)
}

async function fetchExpenses() {
    const { data, error } = await supabase
      .from('expenses')
      .select('*')
      .eq('group_id', params.id) // Filter by groupId
  
    if (error) {
      console.error('Error fetching expenses:', error)
      return
    }
  
    setExpenses(data)
  }
  
  useEffect(() => {
    if (group && expenses.length > 0 && user) {
      calculateTotalPending(expenses)
    }
  }, [group, expenses, user])

  function calculateTotalPending(expenses) {
    const total = expenses.reduce((acc, expense) => {
      if (expense.paid_by !== user.id) {
        return acc + (expense.amount / group?.members.length+1)
      }
      return acc
    }, 0)
    setTotalPending(total)
  }

  async function handleSettleUp() {
    const response = await fetch(`/api/settle-up`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ groupId: params.id })
    })
    if (response.ok) {
      fetchExpenses()
    } else {
      console.error('Error settling up')
    }
  }

  if (!group) return <div>Loading...</div>

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">{group.name}</h1>
      
      <Card className="mb-4">
        <CardHeader>
          <CardTitle>Total Pending: Rs.{totalPending.toFixed(2)}</CardTitle>
        </CardHeader>
        <CardContent>
          <Button onClick={handleSettleUp}>Settle Up</Button>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Expenses</CardTitle>
        </CardHeader>
        <CardContent>
          {expenses.map((expense) => (
            <div key={expense.id} className="mb-2">
              <strong>{expense.description}</strong> - Rs.{expense.amount} 
              (Paid by: {expense.paid_by === user.id ? 'You' : expense.paid_by})
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  )
}