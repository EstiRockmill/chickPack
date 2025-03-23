import { Box } from "./Box.model";
import { Category } from "./Category.model";

export class Item{
    id!:number;
    name!:string;
    capacity!:number;
    boxId!:number;
    box!:Box;
    categoryId!:number;
    category!:Category;
}