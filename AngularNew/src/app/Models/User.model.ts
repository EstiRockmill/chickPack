import { Room } from "./Room.model";

export class User{
    id!:number;
    name!:string;
    email!:string;
    password!:string;
    address!:string;
    rooms!:Room[];
}

