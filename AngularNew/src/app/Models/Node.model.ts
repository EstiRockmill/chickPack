import { Box } from "./Box.model";

export class Node{
    box!:Box;
    parent!:Node;
    left!:Node;
    right!:Node;
    height!:number;
}