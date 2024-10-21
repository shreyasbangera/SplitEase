"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../../components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import { Skeleton } from "@/components/ui/skeleton";

export default function GroupDetails({ params }) {
  const { user } = useAuth();
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
      // Fetch expenses
      const { data: expensesData, error: expensesError } = await supabase
        .from("expenses")
        .select(`*`)
        .eq("group_id", params.id)
        .order("created_at", { ascending: false });

      if (expensesError) throw expensesError;

      // Fetch splits for current user
      const { data: splitsData, error: splitsError } = await supabase
        .from("expense_splits")
        .select("*")
        .eq("user_email", user.email)
        .eq("is_settled", false);

      if (splitsError) throw splitsError;

      setExpenses(expensesData);
      setSplits(splitsData);

      // Calculate total pending amount
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
      {group ? (<div className="py-10 flex-1 justify-center lg:max-w-[50%] max-w-[85%]">
        <h1 className="lg:text-3xl text-2xl font-extrabold py-4">
          {group.name}
        </h1>
        <div className="flex my-3 justify-between py-5 bg-gray-100 px-4 rounded-xl">
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
          {expenses.map((expense) => (
            <div key={expense.id} className="mb-2">
              <div className=" flex justify-between">
              <span className="font-medium">{expense.description}</span> 
              <p>Rs.{expense.amount}</p>
              </div>
              <p>Paid by: {expense.paid_by === user.id ? "You" : expense.paid_by}</p>
            </div>
          ))}
        </div>
        </div>
      </div>)
      :
      (<div className="py-10 flex-1 justify-center lg:max-w-[50%] max-w-[85%]">
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
