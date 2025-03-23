import { Box } from "./Box.model";
import { User } from "./User.model";

export class Room{
    id!:number;
    name!:string;
    numOfBoxes!:number;
    boxes!:Box[];
    userId!:number;
    user!:User;
}