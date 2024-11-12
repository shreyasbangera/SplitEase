"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../../components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import { Skeleton } from "@/components/ui/skeleton";
import { useRequireAuth } from "@/components/useRequireAuth";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";

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
        .eq("group_id", params.id);

      if (splitsError) throw splitsError;

      if (expensesData.length) {
        setExpenses(expensesData);
      } else {
        setExpenses(null);
      }
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

  console.log(splits)

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

  async function handleSettleUp(expenseSplitsId, shareAmount, paidTo) {
    const { error: settlementError } = await supabase
      .from("settlements")
      .insert({
        group_id: params.id,
        paid_by: user.id,
        paid_to: paidTo,
        amount: shareAmount,
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
      .eq("is_settled", false)
      .eq("id", expenseSplitsId);

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
      {group ? (
        <div className="lg:py-10 py-3 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
          <h1 className="lg:text-3xl text-2xl font-bold py-4">{group.name}</h1>
          <div className="flex lg:my-3 justify-between py-5 bg-gray-100 px-4 rounded-xl">
            <div>
              <p className="font-medium text-sm">Total balance</p>
              <p className="font-bold text-xl">Rs.{totalPending.toFixed(2)}</p>
            </div>
            <Dialog>
              <DialogTrigger asChild>
                <Button
                  className="font-bold text-sm px-4 rounded-xl"
                  // onClick={handleSettleUp}
                  disabled={totalPending === 0}
                >
                  Settle Up
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>Settle Up</DialogTitle>
                  <DialogDescription>
                    Make changes to your profile here. Click save when you're
                    done.
                  </DialogDescription>
                </DialogHeader>
                {splits.length ? (<div className="flex flex-col gap-4">
                  {splits.map(({paid_by_name, share_amount, id, paid_by}) => (
                    <div key={id} className="flex justify-between items-center">
                    <span>{paid_by_name}</span>
                    <Button onClick={() => handleSettleUp(id, share_amount, paid_by)}>Pay Rs.{share_amount}</Button>
                    </div>))}
                </div> 
                ) : (
                  <div className="text-center h-10">Settled up</div>
                )}
              </DialogContent>
            </Dialog>
          </div>
          <div className="py-3">
            <p className="text-lg font-bold py-3">Expenses</p>
            {expenses === null ? (
              <div>No Expenses</div>
            ) : expenses?.length ? (
              <div className="flex flex-col gap-4">
                {expenses?.map((expense) => (
                  <div key={expense.id} className="flex items-center gap-4">
                    <svg
                      className="w-12 h-12"
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M4 2v20l2-1 2 1 2-1 2 1 2-1 2 1 2-1 2 1V2l-2 1-2-1-2 1-2-1-2 1-2-1-2 1Z" />
                      <path d="M14 8H8" />
                      <path d="M16 12H8" />
                      <path d="M13 16H8" />
                    </svg>
                    <div className="w-full">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">
                          {expense.description}
                        </span>
                        <p>Rs.{expense.amount}</p>
                      </div>
                      <p>
                        {expense.paid_by === user?.id
                          ? "You"
                          : expense.paid_by_name}{" "}
                        paid
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div>
                {[1, 2, 3].map((index) => (
                  <Skeleton
                    key={index}
                    className="mb-2 flex justify-between h-[48px] lg:h-[54px]"
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="lg:py-10 py-3 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
          <Skeleton className="w-[50%] lg:h-[36px] h-16 mt-4 mb-7" />
          <Skeleton className="flex justify-between py-5 px-4 lg:mb-3 h-[88px]" />
          <div className="py-3">
            <Skeleton className="my-3 w-24 h-7" />
            <div>
              {[1, 2, 3].map((index) => (
                <Skeleton
                  key={index}
                  className="mb-2 flex justify-between h-[48px] lg:h-[54px]"
                />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
