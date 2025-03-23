namespace EasyPack.API.Models.Room
{
    public class RoomOptimize
    {
        public string Name { get; set; }
        public int UserId { get; set; }
        public int CategoryId { get; set; }
        public int NumOfBoxes { get; set; }
        public List<Box> Boxes { get; set; }
    }
}
