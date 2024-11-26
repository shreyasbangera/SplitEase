import { Info } from 'lucide-react'
import React from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './ui/dialog'

const GroupInfo = ({groupMembers}) => {

  const capitalize = (str) => {
    return str[0].toUpperCase() + str.slice(1).toLowerCase()
  }
  return (
        <Dialog>
                <DialogTrigger asChild>
                <Info size={20} className='cursor-pointer' />
                </DialogTrigger>
                <DialogContent className="sm:max-w-[425px]">
                  <DialogHeader>
                    <DialogTitle>Group Info</DialogTitle>
                  </DialogHeader>
                    <ul className='flex flex-col gap-2'>
                      {groupMembers.map(
                        ({ id, user_name }) => ( 
                            <li key={id}>
                            {capitalize(user_name)}
                            </li>
                        )
                      )}
                    </ul>
                </DialogContent>
              </Dialog>
  )
}

export default GroupInfo