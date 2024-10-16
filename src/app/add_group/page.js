"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "../../components/AuthProvider";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { supabase } from "@/lib/supabaseClient";
import emailjs from "@emailjs/browser";

export default function AddGroup() {
  const { user } = useAuth();
  const router = useRouter();
  const [groupName, setGroupName] = useState("");
  const [inviteEmails, setInviteEmails] = useState([]);

  async function handleCreateGroup() { 
    const { data, error } = await supabase
      .from("groups")
      .insert([{ name: groupName, invite_emails: inviteEmails.split(',') }])
      .select()

      console.log(data)

    if (data && data[0]) {
      await sendInvites(data[0]?.id)
    }
    if (error) {
      console.error("Error creating group:", error);
    } else {
      router.push("/");
    }
  }

    const validateEmail = (email) => {
      return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    };

    console.log(user)

    const sendInvites = async (groupId) => {
      console.log(inviteEmails)
      const emails = inviteEmails.split(',')
        .map((email) => email.trim())
        .filter((email) => validateEmail(email));
        const groupLink = process.env.NODE_ENV === 'production'
        ? `https://yourdomain.com/join_group/${groupId}`  
        : `http://localhost:3000/join_group/${groupId}`;

      if (emails.length > 0) {
        setInviteEmails([]);
        for (const email of emails) {
          emailjs.send(
            "service_1uig6lz",
            "template_6dxjwu3",
            {
              to_name: "Viraj",
              to_email: email,
              from_name: "Shreyas",
              message: `Group invite: ${groupLink}` ,
            },
            "AVaP81TyCDlZxN-1L"
          );
        }
      } else {
        alert("Please enter valid email(s)");
      } 
  };


  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Create New Group</h1>

      <Card>
        <CardHeader>
          <CardTitle>Group Details</CardTitle>
        </CardHeader>
        <CardContent>
          <Input
            placeholder="Group Name"
            value={groupName}
            onChange={(e) => setGroupName(e.target.value)}
            className="mb-2"
          />
          <Input
            placeholder="Invite Member (Email)"
            value={inviteEmails}
            onChange={(e) => setInviteEmails(e.target.value)}
            className="mb-2"
          />
          <Button onClick={handleCreateGroup}>Create Group</Button>
        </CardContent>
      </Card>
    </div>
  );
}
