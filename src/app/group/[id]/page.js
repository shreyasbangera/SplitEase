"use client";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../../components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import { Skeleton } from "@/components/ui/skeleton";
import { useRequireAuth } from "@/components/useRequireAuth";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { BookCheck, Coins, IndianRupee, List, Loader2, Scale, Wallet } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

export default function GroupDetails({ params }) {
  const user = useRequireAuth();
  const router = useRouter();
  const [group, setGroup] = useState(null);
  const [expenses, setExpenses] = useState([]);
  const [totalPending, setTotalPending] = useState(0);
  const [settlements, setSettlements] = useState([]);
  const [splits, setSplits] = useState([]);
  const [loadingId, setLoadingId] = useState(null);
  const [unsettledBalance, setUnsettledBalance] = useState([]);
  const { toast } = useToast();

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
    setLoadingId(expenseSplitsId);
    try {
      const { error: settlementError } = await supabase
        .from("settlements")
        .insert({
          group_id: params.id,
          paid_by: user.id,
          paid_to: paidTo,
          amount: shareAmount,
        });

      if (settlementError) throw settlementError;

      const { data, error: splitUpdateError } = await supabase
        .from("expense_splits")
        .update({
          is_settled: true,
          settled_at: new Date().toISOString(),
        })
        .eq("user_email", user.email)
        .eq("is_settled", false);

      if (splitUpdateError) throw splitUpdateError;

      await fetchExpensesAndSplits();
      await fetchSettlements();
    } catch (error) {
      console.error("Error settling up:", error);
    } finally {
      setLoadingId(null);
      toast({ description: "Payment Successfull!" });
    }
  }

  const handleUnsettledBalance = async () => {
    const { data, error } = await supabase
      .from("expense_splits")
      .select("*")
      .eq("is_settled", false)
      .eq("group_id", params.id)
      .eq("paid_by", user.id);

    setUnsettledBalance(data);

    if (error) throw error;
  };

  const groupedSplits = splits.reduce(
    (acc, { share_amount, id, paid_by, paid_by_name }) => {
      const existingUser = acc.find(
        (item) => item.paid_by.user_id === paid_by.user_id
      );

      if (existingUser) {
        existingUser.share_amount += share_amount;
      } else {
        acc.push({ share_amount, id, paid_by, paid_by_name });
      }

      return acc;
    },
    []
  );

  return (
    <div className="flex justify-center">
      {group ? (
        <div className="lg:py-10 py-3 flex-1 justify-center lg:max-w-[60%] max-w-[85%]">
          <div className="flex items-center justify-between">
            <h1 className="lg:text-3xl text-2xl font-bold py-4">
              {group.name}
            </h1>
            <Dialog className="min-h-20">
              <DialogTrigger asChild>
              <div>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Scale
                        strokeWidth={1.5}
                        className="cursor-pointer w-8 h-8"
                        onClick={() => handleUnsettledBalance()}
                      />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p>Unsettled Debts</p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                </div>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>Unsettled Debts</DialogTitle>
                  <div className="py-4">
                    {unsettledBalance.length ? (unsettledBalance?.map(
                      ({ user_email, id, share_amount }) => (
                        <li
                          key={id}
                        >{`${user_email} owes you Rs.${share_amount}`}</li>
                      )
                    )):(
                      <div className="flex justify-center items-center flex-col gap-2 py-4 font-semibold opacity-30"><BookCheck />No debts</div>)}
                  </div>
                </DialogHeader>
              </DialogContent>
            </Dialog>
          </div>
          <div className="flex lg:my-3 justify-between py-5 bg-gray-100 px-4 rounded-xl">
            <div>
              <p className="font-medium text-sm">Total balance</p>
              <p className="font-semibold text-xl">Rs.{totalPending.toFixed(2)}</p>
            </div>
            <Dialog className="min-h-20">
              <DialogTrigger asChild>
                <Button
                  className="font-medium text-sm px-4 rounded-xl"
                  disabled={totalPending === 0}
                >
                  Settle Up
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                  <DialogTitle>Settle Up</DialogTitle>
                </DialogHeader>
                {splits.length ? (
                  <div className="flex flex-col gap-4">
                    {groupedSplits.map(
                      ({ paid_by_name, share_amount, id, paid_by }) => (
                        <div
                          key={id}
                          className="flex justify-between items-center"
                        >
                          <span>{paid_by_name}</span>
                          <Button
                            className="min-w-[98px]"
                            onClick={() =>
                              handleSettleUp(id, share_amount, paid_by)
                            }
                          >
                            {loadingId === id ? (
                              <Loader2 className="animate-spin" />
                            ) : (
                              `Pay Rs.${share_amount}`
                            )}
                          </Button>
                        </div>
                      )
                    )}
                  </div>
                ) : (
                  <div className="min-h-20 flex flex-col gap-2 justify-center items-center py-4 opacity-30">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      height="42px"
                      viewBox="0 -960 960 960"
                      width="42px"
                      fill="#000"
                    >
                      <path d="m424-296 282-282-56-56-226 226-114-114-56 56 170 170Zm56 216q-83 0-156-31.5T197-197q-54-54-85.5-127T80-480q0-83 31.5-156T197-763q54-54 127-85.5T480-880q83 0 156 31.5T763-763q54 54 85.5 127T880-480q0 83-31.5 156T763-197q-54 54-127 85.5T480-80Zm0-80q134 0 227-93t93-227q0-134-93-227t-227-93q-134 0-227 93t-93 227q0 134 93 227t227 93Zm0-320Z" />
                    </svg>
                    <span className="font-semibold">
                      Settled up
                    </span>
                  </div>
                )}
              </DialogContent>
            </Dialog>
          </div>
          <div className="py-3">
            <p className="text-lg font-bold py-3">Expenses</p>
            {expenses === null ? (
              <div className="flex flex-col gap-2 justify-center items-center h-[30vh] font-semibold text-lg opacity-30">
              <Wallet size={42} />
                <span>No Expenses</span>
              </div>
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
