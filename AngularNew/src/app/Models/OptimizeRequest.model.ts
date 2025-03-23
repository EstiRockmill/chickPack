import { RoomPut } from "./RoomPut.model";

export interface OptimizeRequest {
    capacity: number;
    roomId: number;
    itemJsons: { label: string; volumeCm3: number }[];
  }