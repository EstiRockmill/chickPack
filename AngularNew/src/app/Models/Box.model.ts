import { Item } from "./Item.model";
import { Room } from "./Room.model";

export class Box {
    id!:number ;
    contents!:string;
    capacity!:number;
    items!:Item[];
    roomId!:number;
    room!:Room;   
}