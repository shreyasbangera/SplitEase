import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { CreditCard } from "lucide-react";

const Settlements = ({settlements, user}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Settlements</CardTitle>
      </CardHeader>
      <CardContent>
        {settlements.length ? (
          <ul className="space-y-4">
            {settlements.map((settlement) => (
              <li
                key={settlement.id}
                className="flex items-center justify-between pb-2"
              >
                <div className="flex items-center gap-4">
                  <CreditCard size={24} className="text-green-500" />
                  <div>
                    <p className="font-medium">
                      {settlement.paid_by === user.id
                        ? "You"
                        : settlement.paid_by_name}{" "}
                      paid{" "}
                      {settlement.paid_to === user.id
                        ? "you"
                        : settlement.paid_to_name}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {new Date(settlement.settled_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <span className="font-semibold">
                  Rs.{settlement.amount.toFixed(2)}
                </span>
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
  );
};

export default Settlements;
