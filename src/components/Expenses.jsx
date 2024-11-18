import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ReceiptText } from "lucide-react";
import { Skeleton } from "./ui/skeleton";

const Expenses = ({ expenses, user }) => {
  return (
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
          <ul className="flex flex-col gap-4">
            {expenses.map((expense) => (
              <li
                key={expense.id}
                className="flex items-center justify-between pb-2"
              >
                <div className="flex items-center gap-4">
                  <ReceiptText size={24} />
                  <div>
                    <p className="font-medium">{expense.description}</p>
                    <p className="text-sm text-muted-foreground">
                      {expense.paid_by === user?.id
                        ? "You"
                        : expense.paid_by_name}{" "}
                      paid
                    </p>
                  </div>
                </div>
                      <p className="font-semibold">Rs.{expense.amount.toFixed(2)}</p>
              </li>
            ))}
          </ul>
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
      </CardContent>
    </Card>
  );
};

export default Expenses;
