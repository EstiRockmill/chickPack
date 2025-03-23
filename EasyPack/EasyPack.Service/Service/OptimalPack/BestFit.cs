using EasyPack.Core.Models;
using EasyPack.Core.Service;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EasyPack.Service.Service.OptimalPack
{
    public class BestFit : IBestFit
    {
        private readonly IAVLTreeService _treeService;

        public BestFit(IAVLTreeService treeService)
        {
            _treeService = treeService;
        }

        public void findSmallestSpace(Item item, double c, Node root)
        {
            Node bestFit = findBestFitNode(item, root);

            if (bestFit != null)
            {
                updateBoxWithItem(bestFit, item);
            }
            else
            {
                createNewBoxWithItem(item, c, root);
            }
        }

        public Node findBestFitNode(Item item, Node root)
        {
            Node curr = root;
            Node bestFit = null;

            while (curr != null)
            {
                if (curr.Box.Capacity >= item.Capacity)
                {
                    bestFit = curr;
                    curr = curr.Left;
                }
                else
                {
                    curr = curr.Right;
                }
            }

            return bestFit;
        }

        public void updateBoxWithItem(Node boxNode, Item item)
        {
            if (boxNode.Left == null && boxNode.Right == null)
            {
                boxNode.Box.Capacity -= item.Capacity;
                boxNode.Box.Items.Add(item);
            }
            else
            {
                Node deletedNode = _treeService.DeleteAVL(boxNode);

                if (deletedNode == null)
                {
                    throw new InvalidOperationException("Failed to delete node.");
                }
                //deletedNode.Left = null;
                //deletedNode.Right = null;
                //deletedNode.Parent = null;
                deletedNode.Box.Capacity -= item.Capacity;
                deletedNode.Box.Items.Add(item);
                Node treeRoot = getRoot(deletedNode);
                _treeService.AVLInsert(deletedNode.Box, treeRoot);
            }
        }

        public void createNewBoxWithItem(Item item, double c, Node root)
        {
            Box b = new Box(c - item.Capacity);
            b.Items.Add(item);
            Node newNode = new Node(b);
            _treeService.AVLInsert(b, root);
        }

        public Node getRoot(Node node)
        {
            while (node.Parent != null)
            {
                node = node.Parent;
            }
            return node;
        }

        public int bestFitPack(Node root, List<Item> items, double c)
        {
            foreach (Item item in items)
            {
                if (c >= item.Capacity)
                {
                    findSmallestSpace(item, c, getRoot(root));
                }
            }
            return _treeService.Size(getRoot(root));
        }
    }
}
