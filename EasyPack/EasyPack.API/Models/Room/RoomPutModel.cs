namespace EasyPack.API.Models.Room
{
    public class RoomPutModel
    {
        public int Id { get; set; } 
        public string Name { get; set; }
        public int UserId { get; set; }
        public int NumOfBoxes { get; set; }


    }
}
