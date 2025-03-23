using EasyPack.Core.Models;
using EasyPack.Core.Service;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Service.Service.OptimalPack
{
    public class NodeService:INodeService
    {
        private readonly IBestFit _bestFit;
        private readonly IFirstFit _firstFit;
        private readonly IRoomService _roomService;
        public NodeService(IBestFit bestFit,IFirstFit firstFit,IRoomService roomService)
        {
            _bestFit = bestFit;
            _firstFit = firstFit;
            _roomService = roomService;
        }
        public void TraverseInOrder(Node node, List<Box> boxList)
        {
            if (node == null) return;

            TraverseInOrder(node.Left, boxList);
            boxList.Add(node.Box);
            TraverseInOrder(node.Right, boxList);
        }
        public List<Box> ConvertTreeToList(Node root)
        {
            List<Box> boxList = new List<Box>();
            TraverseInOrder(root, boxList);
            return boxList;
        }
        public Node getRoot(Node node)
        {
            while (node.Parent != null)
            {
                node = node.Parent;
            }
            return node;
        }
        public Room compareBetweenTwoWays(List<Item>items,double capacity,Room room)
        {
            Node rootBestFit = new Node(new Box(capacity));
            int boxesBestFit = _bestFit.bestFitPack(rootBestFit, items, capacity);
            rootBestFit=getRoot(rootBestFit);
            Node rootFirstFit = new Node(new Box(capacity));
            int boxesFirstFit=_firstFit.FirstFitPack(rootFirstFit, items, capacity);
            rootFirstFit=getRoot(rootFirstFit);
            if (boxesBestFit >= boxesFirstFit)
            {
                room.NumOfBoxes = boxesFirstFit;
                room.Boxes = ConvertTreeToList(rootFirstFit);
                _roomService.UpdateRoom(room, room.Id);
            }
            else
            {
                room.NumOfBoxes = boxesBestFit;
                room.Boxes=ConvertTreeToList(rootBestFit);
                _roomService.UpdateRoom(room, room.Id);
            }
            return room;
        }
    }
}
