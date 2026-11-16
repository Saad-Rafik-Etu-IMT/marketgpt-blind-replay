import { listInstruments } from "@/lib/api";

// Operator metadata only. Never passed to replay observations or model inputs.
export const loadResearchInventory = listInstruments;
