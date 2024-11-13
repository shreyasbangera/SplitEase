"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "../components/ui/button";
import { supabase } from "@/lib/supabaseClient";
import { useRequireAuth } from "@/components/useRequireAuth";
import Image from "next/image";
import { Skeleton } from "../components/ui/skeleton";
import { useToast } from "@/hooks/use-toast";

export default function Home() {
  const user = useRequireAuth();
  const router = useRouter();
  const [groups, setGroups] = useState([]);
  const [loading, setLoading] = useState(true);
  const { toast } = useToast();

  useEffect(() => {
    if (user) fetchGroups();
  }, [user]);

  async function fetchGroups() {
    const { data, error } = await supabase
      .from("groups")
      .select("*")
      .or(`created_by.eq.${user.id},invited_emails.cs.{${user.email}}`);

    if (error) {
      console.error("Error fetching groups:", error);
    } else {
      setGroups(data);
    }
    setLoading(false);
  }

  console.log(user)

  async function handleAddExpense() {
    if (groups.length > 0) {
      router.push("/add_expense");
    } else {
      toast({
        title: "No Group Found",
        description: "Please create a group first before adding an expense.",
      });
    }
  }

  return (
    <div className="flex justify-center w-full">
      <div className="py-5 lg:max-w-[60%] max-w-[85%] w-full">
        <div className="flex lg:flex-row flex-col justify-end gap-3">
          <Button
            onClick={() => router.push("/create_group")}
            className="font-bold bg-[#e7eef4] text-[#0d151c] text-sm hover:bg-[#e7eef4]/80 px-4 rounded-xl"
          >
            Create new group
          </Button>
          <Button
            className="font-bold text-sm px-4 rounded-xl"
            onClick={handleAddExpense}
          >
            Add new expense
          </Button>
        </div>
        <h1 className="lg:p-4 mt-8 lg:mt-0 p-0 font-bold lg:text-4xl text-2xl tracking-tighter">
          Groups
        </h1>

        {loading ? (
          <div>
            {[1, 2, 3].map((index) => (
              <div
                key={index}
                className="flex items-center py-2 justify-between"
              >
                <div className="py-2 px-0 lg:px-4 font-medium flex items-center gap-4">
                  <Skeleton className="w-14 lg:w-14 h-14 rounded-full" />
                  <Skeleton className="h-6 w-32 lg:w-48" />
                </div>
                <Skeleton className="rounded-xl h-8 w-20" />
              </div>
            ))}
          </div>
        ) : (
          <>
            {groups?.length ? (
              <div>
                {groups.map((group) => (
                  <div
                    key={group.id}
                    onClick={() => router.push(`/group/${group.id}`)}
                    className="flex items-center py-2 justify-between cursor-pointer"
                  >
                    <div className="py-2 px-0 lg:px-4 font-medium flex items-center gap-4 cursor-pointer">
                      <svg
                        className="lg:w-[54px] lg:h-[54px] w-9 h-9"
                        xmlns="http://www.w3.org/2000/svg"
                        width="54"
                        height="54"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="1"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      >
                        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                      </svg>
                      <span>{group.name}</span>
                    </div>
                    <Button className="rounded-xl text-sm font-medium h-8 bg-[#E7EEF4] text-[#0d151c] hover:bg-[#E7EEF4]/80">
                      Settle
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col gap-2 justify-center items-center custom-height">
                <span className="font-bold text-lg">
                  You haven't created any group yet
                </span>
                <span className="text-[#0d151c] text-sm text-center">
                  Create a group to share expenses with friends, family or
                  roommates
                </span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
