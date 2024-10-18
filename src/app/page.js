"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../components/AuthProvider";
import { Button } from "../components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "../components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import { useRequireAuth } from "@/components/useRequireAuth";
import Image from "next/image";

export default function Home() {
  const user = useRequireAuth();
  const router = useRouter();
  const [groups, setGroups] = useState([]);

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
  }

  async function handleAddExpense() {
    if (groups.length > 0) {
      router.push("/add_expense");
    } else {
      alert("Please create a group first before adding an expense.");
    }
  }

  return (
    <div className="flex justify-center w-full">
    <div className="py-5 lg:max-w-[70%] max-w-[85%] w-full">
      <div className="flex lg:flex-row flex-col justify-end gap-3">
        <Button
          onClick={() => router.push("/add_group")}
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
      <h1 className="lg:p-4 mt-8 lg:mt-0 p-0 font-extrabold lg:text-4xl text-2xl tracking-tighter">Groups</h1>
      {groups?.length ? (
        <div>
          {groups.map((group) => (
            <div key={group.id} onClick={() => router.push(`/group/${group.id}`)} className="flex items-center py-2 justify-between cursor-pointer">
              <div
                className="py-2 px-0 lg:px-4 font-medium flex items-center gap-4 cursor-pointer"
              >
                <Image className="w-9 lg:w-14" src="https://www.svgrepo.com/show/86044/group.svg" width={56} height={56} alt='group' />
                <span>{group.name}</span>
              </div>
              <Button className='rounded-xl text-sm font-medium h-8 bg-[#E7EEF4] text-[#0d151c] hover:bg-[#E7EEF4]/80'>Settle</Button>
            </div>
          ))}
        </div>
      ) : (
        <div className="flex flex-col gap-2 justify-center items-center custom-height">
          <span className="font-bold text-lg">
            You haven't created any group yet
          </span>
          <span className="text-[#0d151c] text-sm">
            Create a group to share expenses with friends, family or roommates
          </span>
        </div>
      )}
    </div>
    </div>
  );
}
