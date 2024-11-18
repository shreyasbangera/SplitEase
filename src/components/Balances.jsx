import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ArrowLeft, ArrowRight, CheckCircle } from "lucide-react";

const Balances = ({ unsettledDebts, groupedSplits }) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Balances</CardTitle>
      </CardHeader>
      <CardContent>
        {unsettledDebts.length ? (
          <div>
            <p className="pb-2 font-semibold">They owe</p>
            <ul className="space-y-2">
              {unsettledDebts.map(({ user_email, id, share_amount }) =>
                share_amount > 0 ? (
                  <li key={id} className="flex justify-between items-center">
                  <div className="flex items-center gap-4">
                  <ArrowRight className="text-green-500" />
                  <div>
                    <span>{user_email}</span>
                    </div>
                    </div>
                    <span className="font-semibold">
                      Rs.{share_amount.toFixed(2)}
                    </span>
                  </li>
                ) : null
              )}
            </ul>
            <p className="pb-2 pt-4 font-semibold">You owe</p>
            <ul className="space-y-2">
              {groupedSplits.map(({ paid_by_name, share_amount, id}) =>
                share_amount > 0 ? (
                  <li key={id} className="flex justify-between items-center">
                  <div className="flex items-center gap-4">
                  <ArrowLeft className="text-red-500" />
                  <div>
                    <span>{paid_by_name}</span>
                    </div>
                    </div>
                    <span className="font-semibold">
                      Rs.{share_amount.toFixed(2)}
                    </span>
                  </li>
                ) : null
              )}
            </ul>
          </div>
        ) : (
          <div className="flex justify-center items-center h-[30vh] flex-col gap-2 py-4 font-semibold opacity-30">
            <CheckCircle size={42} />
            <span>No Balances</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default Balances;
