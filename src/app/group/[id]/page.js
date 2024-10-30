"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../../components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import { Skeleton } from "@/components/ui/skeleton";
import { useRequireAuth } from "@/components/useRequireAuth";

export default function GroupDetails({ params }) {
  const user = useRequireAuth();
  const router = useRouter();
  const [group, setGroup] = useState(null);
  const [expenses, setExpenses] = useState([]);
  const [totalPending, setTotalPending] = useState(0);
  const [settlements, setSettlements] = useState([]);
  const [splits, setSplits] = useState([]);

  useEffect(() => {
    if (user && params.id) {
      fetchGroup();
      fetchExpensesAndSplits();
    }
  }, [user, params.id]);

  async function fetchGroup() {
    const { data, error } = await supabase
      .from("groups")
      .select("*")
      .eq("id", params.id)
      .single();

    if (error) {
      console.error("Error fetching group:", error);
      return;
    }
    setGroup(data);
  }

  const fetchExpensesAndSplits = async () => {
    try {
      const { data: expensesData, error: expensesError } = await supabase
        .from("expenses")
        .select(`*`)
        .eq("group_id", params.id)
        .order("created_at", { ascending: false });

      if (expensesError) throw expensesError;

      const { data: splitsData, error: splitsError } = await supabase
        .from("expense_splits")
        .select("*")
        .eq("user_email", user.email)
        .eq("is_settled", false)
        .eq("group_id", params.id)

      if (splitsError) throw splitsError;

      setExpenses(expensesData);
      setSplits(splitsData);

      const pending = splitsData.reduce(
        (sum, split) => sum + split.share_amount,
        0
      );
      setTotalPending(pending);
    } catch (error) {
      console.error(error);
    }
  };


  const fetchSettlements = async () => {
    const { data, error } = await supabase
      .from("settlements")
      .select("*")
      .eq("group_id", params.id)
      .order("settled_at", { ascending: false });

    if (error) {
      console.error("Error fetching settlements:", error);
      return;
    }

    setSettlements(data);
  };

  async function handleSettleUp() {
    // First, upsert the settlement
    const { error: settlementError } = await supabase
      .from("settlements")
      .insert({
        group_id: params.id,
        paid_by: user.id,
        amount: totalPending,
      });

    if (settlementError) {
      console.error("Error settling up:", settlementError);
      return;
    }

    const { data, error: splitUpdateError } = await supabase
      .from("expense_splits")
      .update({
        is_settled: true,
        settled_at: new Date().toISOString(),
      })
      .eq("user_email", user.email)
      .eq("is_settled", false);

    if (splitUpdateError) {
      console.error(
        "Error updating settled amounts in expenses:",
        splitUpdateError
      );
      return;
    }
    await fetchExpensesAndSplits();
    await fetchSettlements();
  }


  return (
    <div className="flex justify-center">
      {group ? (<div className="lg:py-10 py-3 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
        <h1 className="lg:text-3xl text-2xl font-bold py-4">
          {group.name}
        </h1>
        <div className="flex lg:my-3 justify-between py-5 bg-gray-100 px-4 rounded-xl">
        <div>
          <p className="font-medium text-sm">Total balance</p>
          <p className="font-bold text-xl">Rs.{totalPending.toFixed(2)}</p>
          </div>
          <Button className='font-bold text-sm px-4 rounded-xl' onClick={handleSettleUp} disabled={totalPending === 0}>
            Settle Up
          </Button>
        </div>
        <div className="py-3">
        <p className="text-lg font-bold py-3">Expenses</p>
        <div>
          {expenses?.map((expense) => (
            <div key={expense.id} className="mb-2 flex items-center gap-4">
            <svg className="lg:w-[54px] lg:h-[54px] w-12 h-12" xmlns="http://www.w3.org/2000/svg"  width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"><path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1-2 1-2-1-2 1Z"/><path d="M14 8H8"/><path d="M16 12H8"/><path d="M13 16H8"/></svg>
              <div className="w-full">
              <div className="flex justify-between items-center">
              <span className="font-medium">{expense.description}</span> 
              <p>Rs.{expense.amount}</p>
              </div>
              <p>{expense.paid_by === user.id ? "You" : expense.paid_by_name} paid</p>
              </div>
            </div>
          ))}
        </div>
        </div>
      </div>)
      :
      (<div className="py-10 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
        <Skeleton className="w-[50%] lg:h-[68px] h-16 py-4" />
        <div className="flex justify-between py-5 px-4 my-3">
        <div>
          <Skeleton className="h-5 w-24" />
          <Skeleton className="h-7 w-24"/>
          </div>
          <Skeleton className='w-24 h-10 rounded-xl'/>
          </div>
          <div className="py-3">
        <Skeleton className="my-3 w-24 h-8"/>
        <div>
          {[1,2,3].map((index) => (
            <div key={index} className="mb-2">
              <div className=" flex justify-between">
              <Skeleton className="w-24 h-4" /> 
              <Skeleton className="w-24 h-4"/>
              </div>
              <Skeleton className="w-28 h-6"/>
            </div>
          ))}
        </div>
        </div>
        </div>)}
    </div>
  );
}
