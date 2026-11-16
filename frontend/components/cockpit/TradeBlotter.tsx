"use client";

import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import { useState } from "react";

import { Button, ConfirmDialog, useToast } from "@/components/ui";
import type { OrderRead } from "@/lib/types";

interface Props {
  orders: OrderRead[];
  onCancel: (orderId: string) => Promise<void> | void;
}

export function TradeBlotter({ orders, onCancel }: Props) {
  const [sorting, setSorting] = useState<SortingState>([
    { id: "submitted_at_bar", desc: true },
  ]);
  const [cancelId, setCancelId] = useState<string | null>(null);
  const toast = useToast();

  const columns: ColumnDef<OrderRead>[] = [
    { accessorKey: "submitted_at_bar", header: "Bar" },
    { accessorKey: "side", header: "Side" },
    { accessorKey: "order_type", header: "Type" },
    { accessorKey: "quantity", header: "Qty" },
    {
      accessorKey: "limit_price",
      header: "Limit",
      cell: (info) => info.getValue<number | null>() ?? "—",
    },
    {
      accessorKey: "stop_price",
      header: "Stop",
      cell: (info) => info.getValue<number | null>() ?? "—",
    },
    { accessorKey: "status", header: "Status" },
    {
      id: "actions",
      header: "",
      cell: ({ row }) =>
        row.original.status === "pending" ? (
          <Button
            variant="danger"
            size="sm"
            aria-label={`Cancel order ${row.original.order_id.slice(0, 8)}`}
            onClick={() => setCancelId(row.original.order_id)}
          >
            Cancel
          </Button>
        ) : null,
    },
  ];

  const table = useReactTable({
    data: orders,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  return (
    <div className="bg-slate-900 rounded border border-slate-800 text-sm">
      <div className="px-3 py-2 font-semibold text-slate-200">Trade blotter</div>
      <table className="w-full text-xs">
        <thead className="text-slate-400 bg-slate-950">
          {table.getHeaderGroups().map((group) => (
            <tr key={group.id}>
              {group.headers.map((header) => (
                <th
                  key={header.id}
                  className="text-left px-2 py-1 cursor-pointer"
                  onClick={header.column.getToggleSortingHandler()}
                >
                  {flexRender(
                    header.column.columnDef.header,
                    header.getContext(),
                  )}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {table.getRowModel().rows.map((row) => (
            <tr key={row.id} className="border-t border-slate-800">
              {row.getVisibleCells().map((cell) => (
                <td key={cell.id} className="px-2 py-1 font-mono">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </td>
              ))}
            </tr>
          ))}
          {orders.length === 0 && (
            <tr>
              <td
                colSpan={columns.length}
                className="px-2 py-3 text-center text-slate-500"
              >
                No orders yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      <ConfirmDialog
        open={cancelId !== null}
        title="Cancel this order?"
        description="The pending order will be withdrawn and will not fill."
        destructive
        confirmLabel="Cancel order"
        cancelLabel="Keep order"
        onConfirm={async () => {
          const id = cancelId;
          setCancelId(null);
          if (!id) return;
          try {
            await onCancel(id);
            toast.success("Order cancelled.");
          } catch (e) {
            toast.error(e instanceof Error ? e.message : "Cancel failed");
          }
        }}
        onCancel={() => setCancelId(null)}
      />
    </div>
  );
}
