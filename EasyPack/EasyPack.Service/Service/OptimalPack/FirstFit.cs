using EasyPack.Core.Models;
using EasyPack.Core.Service;
using EasyPack.Service.Service.OptimalPack;

public class FirstFit:IFirstFit
{
    private readonly IAVLTreeService _avlTreeService;
    public FirstFit(IAVLTreeService avlTreeService)
    {
        _avlTreeService = avlTreeService;
    }

    public void FindFirstPlace(Item item, double c, Node root)
    {
        Node head= getRoot(root);

        while (root != null)
        {
            if (root.Box.Capacity >= item.Capacity)
            {
                root.Box.Capacity -= item.Capacity;
                root.Box.Items.Add(item);

              Node del=_avlTreeService.DeleteAVL(root);
                _avlTreeService.AVLInsert(del.Box, head);

                return;
            }
            if (root.Right != null)
            {
                root = root.Right;
            }
            else
            {
                break;
            }
        }

        Box newBox = new Box(c - item.Capacity);
        newBox.Items.Add(item);
        Node newNode = new Node(newBox);

        _avlTreeService.AVLInsert(newBox, head);
    }
    public Node getRoot(Node node)
    {
        while (node.Parent != null)
        {
            node = node.Parent;
        }
        return node;
    }

    public int FirstFitPack(Node root, List<Item>items, double c)
    {
        foreach(Item item in items)
        {
            if (c >= item.Capacity)
            {
                FindFirstPlace(item, c, getRoot(root));
            }
        }

        return _avlTreeService.Size(getRoot(root));//num of boxes
    }

}