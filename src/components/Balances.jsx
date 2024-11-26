import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ArrowLeft, ArrowRight, CheckCircle } from 'lucide-react';

const Balances = ({ unsettledDebts, groupedSplits }) => {
  const theyOwe = unsettledDebts.filter(({ share_amount }) => share_amount > 0);
  const youOwe = groupedSplits.filter(({ share_amount }) => share_amount > 0);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Balances</CardTitle>
      </CardHeader>
      <CardContent>
        {theyOwe.length || youOwe.length ? (
          <div className="flex flex-col gap-4">
            {theyOwe.length > 0 && (
              <div>
                <p className="pb-2 font-semibold">They owe</p>
                <ul className="space-y-2">
                  {theyOwe.map(({ user_email, id, share_amount }) => (
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
                  ))}
                </ul>
              </div>
            )}
            {youOwe.length > 0 && (
              <div>
                <p className="pb-2 font-semibold">You owe</p>
                <ul className="space-y-2">
                  {youOwe.map(({ paid_by_name, share_amount, id }) => (
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
                  ))}
                </ul>
              </div>
            )}
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

