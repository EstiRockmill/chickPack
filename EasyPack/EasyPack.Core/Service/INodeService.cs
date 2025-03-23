using EasyPack.Core.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Core.Service
{
    public interface INodeService
    {
        public Room compareBetweenTwoWays(List<Item> items, double capacity, Room room);
        public List<Box> ConvertTreeToList(Node root);
        public void TraverseInOrder(Node node, List<Box> boxList);
    }
}
