'use client'

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "../../../components/AuthProvider"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { supabase } from "@/lib/supabaseClient"
import { Skeleton } from "@/components/ui/skeleton"
import { useRequireAuth } from "@/components/useRequireAuth"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { CheckCircle, Loader2, PieChart, Receipt, CreditCard, ReceiptText } from 'lucide-react'
import { useToast } from "@/hooks/use-toast"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"

export default function GroupDetails({ params }) {
  const user = useRequireAuth()
  const router = useRouter()
  const [group, setGroup] = useState(null)
  const [expenses, setExpenses] = useState([])
  const [totalPending, setTotalPending] = useState(0)
  const [settlements, setSettlements] = useState([])
  const [splits, setSplits] = useState([])
  const [loadingId, setLoadingId] = useState(null)
  const [unsettledBalance, setUnsettledBalance] = useState([])
  const [groupedSplits, setGroupedSplits] = useState([])
  const [unsettledDebts, setUnsettledDebts] = useState([])
  const { toast } = useToast()

  useEffect(() => {
    if (user && params.id) {
      fetchGroup()
      fetchExpensesAndSplits()
      fetchSettlements()
    }
  }, [user, params.id])

  async function fetchGroup() {
    const { data, error } = await supabase
      .from("groups")
      .select("*")
      .eq("id", params.id)
      .single()

    if (error) {
      console.error("Error fetching group:", error)
      return
    }
    setGroup(data)
  }

  const fetchExpensesAndSplits = async () => {
    try {
      const { data: expensesData, error: expensesError } = await supabase
        .from("expenses")
        .select(`*`)
        .eq("group_id", params.id)
        .order("created_at", { ascending: false })

      if (expensesError) throw expensesError

      const { data: splitsData, error: splitsError } = await supabase
        .from("expense_splits")
        .select("*")
        .eq("user_email", user.email)
        .eq("is_settled", false)
        .eq("group_id", params.id)

      if (splitsError) throw splitsError

      const { data: unsettledData, error: unsettledError } = await supabase
        .from("expense_splits")
        .select("*")
        .eq("is_settled", false)
        .eq("group_id", params.id)
        .eq("paid_by", user.id)

      if (unsettledError) throw unsettledError

      if (expensesData?.length) {
        setExpenses(expensesData)
      } else {
        setExpenses(null)
      }
      setSplits(splitsData || [])
      setUnsettledBalance(unsettledData || [])

      const groupedSplits = (splitsData || []).reduce(
        (acc, { share_amount, id, paid_by, paid_by_name }) => {
          const existingUser = acc.find((item) => item.paid_by === paid_by)
          if (existingUser) {
            existingUser.share_amount += share_amount
          } else {
            acc.push({ share_amount, id, paid_by, paid_by_name })
          }
          return acc
        },
        []
      )
      setGroupedSplits(groupedSplits)

      const unsettledDebts = (unsettledData || []).reduce(
        (acc, { user_email, id, share_amount, user_id }) => {
          const existingUser = acc.find((item) => item.user_email === user_email)
          if (existingUser) {
            existingUser.share_amount += share_amount
          } else {
            acc.push({ user_email, id, share_amount, user_id })
          }
          return acc
        },
        []
      )
      setUnsettledDebts(unsettledDebts)

      let pending = 0
      for (const gSplit of groupedSplits) {
        const matchingDebt = unsettledDebts.find((debt) => debt.user_id === gSplit.paid_by)
        if (matchingDebt) {
          const netAmount = gSplit.share_amount - matchingDebt.share_amount
          pending += netAmount > 0 ? netAmount : 0
        } else {
          pending += gSplit.share_amount
        }
      }

      setTotalPending(pending)
    } catch (error) {
      console.error(error)
    }
  }

  const fetchSettlements = async () => {
    const { data, error } = await supabase
      .from("settlements")
      .select("*")
      .eq("group_id", params.id)
      .order("settled_at", { ascending: false })

    if (error) {
      console.error("Error fetching settlements:", error)
      return
    }

    setSettlements(data || [])
  }

  async function handleSettleUp(expenseSplitsId, shareAmount, paidTo) {
    setLoadingId(expenseSplitsId)
    try {
      await createSettlement(params.id, user.id, paidTo, shareAmount)

      const { error: splitUpdateError } = await supabase
        .from("expense_splits")
        .update({
          is_settled: true,
          settled_at: new Date().toISOString(),
        })
        .eq("user_email", user.email)
        .eq("is_settled", false)

      if (splitUpdateError) throw splitUpdateError

      await fetchExpensesAndSplits()
      await fetchSettlements()
      toast({ description: "Payment Successful!" })
    } catch (error) {
      console.error("Error settling up:", error)
      toast({ description: "Payment failed. Please try again.", variant: "destructive" })
    } finally {
      setLoadingId(null)
    }
  }

  async function createSettlement(groupId, paidById, paidToId, amount) {
    const { data, error } = await supabase
      .rpc('create_settlement', {
        p_group_id: groupId,
        p_paid_by: paidById,
        p_paid_to: paidToId,
        p_amount: amount
      });
  
    if (error) {
      console.error('Error creating settlement:', error)
      throw error
    }
  
    return data
  }

  return (
    <div className="flex justify-center">
      {group ? (
        <div className="lg:py-10 py-4 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
          <div className="flex items-center justify-between lg:mb-6 mb-4">
            <h1 className="lg:text-3xl text-2xl font-bold">{group.name}</h1>
          </div>
          <Card className="mb-6">
            <CardContent className="flex justify-between items-center p-6">
              <div>
                <p className="font-medium text-sm">Total balance</p>
                <p className="font-semibold text-xl">Rs.{totalPending.toFixed(2)}</p>
              </div>
              <Dialog>
                <DialogTrigger asChild>
                  <Button className="font-medium text-sm px-4 rounded-xl" disabled={totalPending === 0}>
                    Settle Up
                  </Button>
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px]">
                  <DialogHeader>
                    <DialogTitle>Settle Up</DialogTitle>
                  </DialogHeader>
                  {groupedSplits.length ? (
                    <div className="flex flex-col gap-4">
                      {groupedSplits.map(
                        ({ paid_by_name, share_amount, id, paid_by }) => (
                          share_amount > 0 ? (
                            <div key={id} className="flex justify-between items-center">
                              <span>{paid_by_name}</span>
                              <Button
                                className="min-w-[98px]"
                                onClick={() => handleSettleUp(id, share_amount, paid_by)}
                              >
                                {loadingId === id ? (
                                  <Loader2 className="animate-spin" />
                                ) : (
                                  `Pay Rs.${share_amount.toFixed(2)}`
                                )}
                              </Button>
                            </div>
                          ) : null
                        )
                      )}
                    </div>
                  ) : (
                    <div className="min-h-20 flex flex-col gap-2 justify-center items-center py-4 opacity-30">
                      <CheckCircle size={42} />
                      <span className="font-semibold">All settled up</span>
                    </div>
                  )}
                </DialogContent>
              </Dialog>
            </CardContent>
          </Card>
          <Tabs defaultValue="expenses" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="expenses">Expenses</TabsTrigger>
              <TabsTrigger value="unsettled">Debts</TabsTrigger>
              <TabsTrigger value="activity">Settlements</TabsTrigger>
            </TabsList>
            <TabsContent value="expenses">
              <Card>
                <CardHeader>
                  <CardTitle>Expenses</CardTitle>
                </CardHeader>
                <CardContent>
                  {expenses === null ? (
                    <div className="flex flex-col gap-2 justify-center items-center h-[30vh] font-semibold text-lg opacity-30">
                      <ReceiptText size={42} />
                      <span>No Expenses</span>
                    </div>
                  ) : expenses?.length ? (
                    <div className="flex flex-col gap-4">
                      {expenses.map((expense) => (
                        <div key={expense.id} className="flex items-center gap-4">
                          <ReceiptText className="w-12 h-12" />
                          <div className="w-full">
                            <div className="flex justify-between items-center">
                              <span className="font-medium">{expense.description}</span>
                              <p>Rs.{expense.amount.toFixed(2)}</p>
                            </div>
                            <p>{expense.paid_by === user?.id ? "You" : expense.paid_by_name} paid</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div>
                      {[1, 2, 3].map((index) => (
                        <Skeleton key={index} className="mb-2 flex justify-between h-[48px] lg:h-[54px]" />
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="unsettled">
              <Card>
                <CardHeader>
                  <CardTitle>Debts</CardTitle>
                </CardHeader>
                <CardContent>
                  {unsettledDebts.length ? (
                    <ul className="space-y-2">
                      {unsettledDebts.map(({ user_email, id, share_amount }) => (
                        share_amount > 0 ? (
                          <li key={id} className="flex justify-between items-center">
                            <span>{user_email}</span>
                            <span className="font-semibold">Rs.{share_amount.toFixed(2)}</span>
                          </li>
                        ) : null
                      ))}
                    </ul>
                  ) : (
                    <div className="flex justify-center items-center h-[30vh] flex-col gap-2 py-4 font-semibold opacity-30">
                      <CheckCircle size={42} />
                      <span>No debts</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
            <TabsContent value="activity">
              <Card>
                <CardHeader>
                  <CardTitle>Settlements</CardTitle>
                </CardHeader>
                <CardContent>
                  {settlements.length ? (
                    <ul className="space-y-4">
                      {settlements.map((settlement) => (
                        <li key={settlement.id} className="flex items-center justify-between border-b pb-2">
                          <div className="flex items-center gap-2">
                            <CreditCard size={20} className="text-green-500" />
                            <div>
                              <p className="font-medium">
                                {settlement.paid_by === user.id ? 'You' : settlement.paid_by_name} paid {settlement.paid_to === user.id ? 'you' : settlement.paid_to_name}
                              </p>
                              <p className="text-sm text-muted-foreground">
                                {new Date(settlement.settled_at).toLocaleDateString()}
                              </p>
                            </div>
                          </div>
                          <span className="font-semibold">Rs.{settlement.amount.toFixed(2)}</span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="flex justify-center items-center h-[30vh] flex-col gap-2 py-4 font-semibold opacity-30">
                      <CreditCard size={42} />
                      <span>No Settlements</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      ) : (
        <div className="lg:py-10 py-3 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
          <Skeleton className="w-[50%] lg:h-[36px] h-16 mt-4 mb-7" />
          <Skeleton className="flex justify-between py-5 px-4 lg:mb-3 h-[88px]" />
          <div className="py-3">
            <Skeleton className="my-3 w-24 h-7" />
            <div>
              {[1, 2, 3].map((index) => (
                <Skeleton key={index} className="mb-2 flex justify-between h-[48px] lg:h-[54px]" />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}