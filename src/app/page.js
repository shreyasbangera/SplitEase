'use client'

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { supabase } from "@/lib/supabaseClient"
import { useRequireAuth } from "@/components/useRequireAuth"
import { Skeleton } from "@/components/ui/skeleton"
import { useToast } from "@/hooks/use-toast"
import { Plus, Users, ArrowRight } from 'lucide-react'

export default function Home() {
  const user = useRequireAuth()
  const router = useRouter()
  const [groups, setGroups] = useState([])
  const [loading, setLoading] = useState(true)
  const { toast } = useToast()

  useEffect(() => {
    if (user) fetchGroups()
  }, [user])

  async function fetchGroups() {
    const { data, error } = await supabase
      .from("groups")
      .select("*")
      .or(`created_by.eq.${user.id},invited_emails.cs.{${user.email}}`)

    if (error) {
      console.error("Error fetching groups:", error)
    } else {
      setGroups(data)
    }
    setLoading(false)
  }

  async function handleAddExpense() {
    if (groups.length > 0) {
      router.push("/add_expense")
    } else {
      toast({
        title: "No Group Found",
        description: "Please create a group first before adding an expense.",
        variant: "destructive",
      })
    }
  }

  return (
    <div className="container mx-auto px-4 lg:px-10 py-8">
      <div className="flex flex-col space-y-8">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-4">
          <h1 className="text-3xl font-bold tracking-tight">Your Groups</h1>
          <div className="flex space-x-4">
            <Button
              onClick={() => router.push("/create_group")}
              variant="outline"
              className="font-medium"
            >
              <Plus className="mr-2 h-4 w-4" /> Create Group
            </Button>
            <Button
              className="font-medium"
              onClick={handleAddExpense}
            >
              <Plus className="mr-2 h-4 w-4" /> Add Expense
            </Button>
          </div>
        </div>

        {loading ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[1, 2, 3].map((index) => (
              <Card key={index}>
                <CardContent className="p-6">
                  <div className="flex items-center space-x-4">
                    <Skeleton className="h-12 w-12 rounded-full" />
                    <div className="space-y-2">
                      <Skeleton className="h-4 w-[150px]" />
                      <Skeleton className="h-4 w-[100px]" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : groups?.length ? (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {groups.map((group) => (
              <Card
                key={group.id}
                className="transition-shadow hover:shadow-md cursor-pointer"
                onClick={() => router.push(`/group/${group.id}`)}
              >
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className="bg-primary/10 p-2 rounded-full">
                        <Users className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <h2 className="font-semibold text-lg">{group.name}</h2>
                        <p className="text-sm text-muted-foreground">
                          {group.invited_emails?.length || 0} members
                        </p>
                      </div>
                    </div>
                    <ArrowRight className="h-5 w-5 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <Card>
            <CardContent className="p-6 text-center">
              <Users className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              <h2 className="font-semibold text-xl mb-2">No groups yet</h2>
              <p className="text-muted-foreground mb-4">
                Create a group to start sharing expenses with friends, family, or roommates.
              </p>
              <Button onClick={() => router.push("/create_group")}>
                Create Your First Group
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}