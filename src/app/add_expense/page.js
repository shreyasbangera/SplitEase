"use client"
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from "../../components/ui/button"
import { Input } from "../../components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../../components/ui/select"
import { Card, CardHeader, CardTitle, CardContent } from "../../components/ui/card"
import { supabase } from '../../lib/supabaseClient'
import { useAuth } from '@/components/AuthProvider'
import { useToast } from '@/hooks/use-toast'

export default function AddExpensePage() {
  const [description, setDescription] = useState('')
  const [amount, setAmount] = useState('')
  const [groupId, setGroupId] = useState('')
  const [groups, setGroups] = useState([])
  const { toast } = useToast()
  const { user } = useAuth()

  useEffect(() => {
    if(user){
    fetchGroups()
    }
  }, [user])

  async function fetchGroups() {
    const { data, error } = await supabase
      .from('groups')
      .select('*')
      .or(`invited_emails.cs.{${user.email}},created_by.eq.${user.id}`);
    
    if (error) {
      console.error('Error fetching groups:', error)
    } else if (data) {
      setGroups(data)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const { data: expense, error: expenseError } = await supabase
        .from('expenses')
        .insert({
          group_id: groupId,
          description,
          amount: parseFloat(amount),
          paid_by: user.id,
          paid_by_name: user.user_metadata.name
        })
        .select()
        .single();

      if (expenseError) throw expenseError;

      const selectedGroup = groups.find(group => group.id === groupId)

      const splits = selectedGroup?.invited_emails
      ?.filter(email => email !== user.email)
      .map(email => ({
        expense_id: expense.id,
        user_email: email,
        share_amount: expense.amount / selectedGroup.invited_emails.length,
        is_settled: false,
        group_id: groupId,
        paid_by: user.id,
        paid_by_name: user.user_metadata.name
      }));
    
      const { error: splitsError } = await supabase
        .from('expense_splits')
        .insert(splits);

      if (splitsError) throw splitsError;
      toast({description: `Added ${description}!`})
      setDescription('');
      setAmount('');

    } catch (error) {
      console.error('Error adding expense:', error);
    }
  };


  return (
    <div className="flex justify-center">
    <div className="py-10 flex-1 justify-center lg:max-w-[50%] max-w-[85%]">
      <h1 className="lg:text-3xl text-2xl font-extrabold py-4">Add new expense</h1>
          <form onSubmit={handleSubmit}>
          <div className='py-3'>
          <p className='pb-2 font-medium text-base'>Description</p>
            <Input
              placeholder="Ex: Dinner at Pizzeria"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="p-[15px] h-14 rounded-xl"
            />
            </div>
            <div className='py-3'>
            <p className='pb-2 font-medium text-base'>Amount</p>
            <Input
              type="number"
              placeholder="Amount"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="p-[15px] h-14 rounded-xl"
            />
            </div>
            <div className='py-3'>
            <p className='pb-2 font-medium text-base'>Group</p>
            <Select value={groupId} onValueChange={setGroupId} >
              <SelectTrigger className="p-[15px] h-14 rounded-xl">
              <SelectValue placeholder="Select a group" />
              </SelectTrigger>
              <SelectContent>
                {groups.map(({id, name}) => (
                  <SelectItem key={id} value={String(id)}>{name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            </div>
            <Button type="submit" className="w-full mt-3">Add Expense</Button>
          </form>
          </div>
    </div>
  )
}